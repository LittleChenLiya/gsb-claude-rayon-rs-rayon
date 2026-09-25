use crate::iter::plumbing::*;
use crate::iter::*;

use std::fmt::{self, Debug};

/// Boundary policy for ParallelSlice::par_map_windows.
///
/// This controls how windows are formed for center elements that are too
/// close to either end of the slice for a complete neighborhood of left
/// and right elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowBoundary {
    /// Truncate windows at the slice boundaries.
    ///
    /// Every element of the slice produces exactly one output. Windows near
    /// the edges are simply shorter, and WindowCenter::position is adjusted
    /// accordingly, so the closure can tell how much of the neighborhood is
    /// missing.
    Truncate,
    /// Skip centers whose complete window does not fit in the slice.
    ///
    /// Centers with fewer than left elements before them or fewer than
    /// right elements after them produce no output at all. Every window the
    /// closure sees has the full left + 1 + right elements, and
    /// WindowCenter::index still refers to the original slice, so the first
    /// output is centered at index left, not 0.
    Skip,
}

/// Describes the center element of a window passed to the closure of
/// ParallelSlice::par_map_windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowCenter {
    /// The index of the center element in the **original** slice.
    ///
    /// This is never renumbered: with WindowBoundary::Skip the first
    /// reported index is left, and truncated edge windows still report
    /// their true position in the original slice.
    pub index: usize,
    /// The offset of the center element within the window slice passed to
    /// the closure, i.e. window[position] is the center element.
    ///
    /// This equals left for every complete window, and is smaller near the
    /// start of the slice when WindowBoundary::Truncate shortens a window.
    pub position: usize,
}

/// Parallel iterator that maps each element of a slice (the "center") to an
/// output computed from a read-only window of its neighborhood.
///
/// This struct is created by the par_map_windows() method on ParallelSlice.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
#[derive(Clone)]
pub struct MapWindows<'data, T, F> {
    slice: &'data [T],
    left: usize,
    right: usize,
    boundary: WindowBoundary,
    map_op: F,
}

impl<T: Debug, F> Debug for MapWindows<'_, T, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MapWindows")
            .field("slice", &self.slice)
            .field("left", &self.left)
            .field("right", &self.right)
            .field("boundary", &self.boundary)
            .finish()
    }
}

impl<'data, T, F> MapWindows<'data, T, F> {
    pub(super) fn new(
        slice: &'data [T],
        left: usize,
        right: usize,
        boundary: WindowBoundary,
        map_op: F,
    ) -> Self {
        // Reject a neighborhood whose total width cannot even be counted.
        left.checked_add(right)
            .expect("par_map_windows: left + right neighborhood width overflowed");
        MapWindows {
            slice,
            left,
            right,
            boundary,
            map_op,
        }
    }
}

fn centers_len(boundary: WindowBoundary, len: usize, left: usize, right: usize) -> usize {
    match boundary {
        WindowBoundary::Truncate => len,
        WindowBoundary::Skip => len.saturating_sub(left + right),
    }
}

impl<'data, T, F, R> ParallelIterator for MapWindows<'data, T, F>
where
    T: Sync,
    F: Fn(&'data [T], WindowCenter) -> R + Sync + Send,
    R: Send,
{
    type Item = R;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(centers_len(
            self.boundary,
            self.slice.len(),
            self.left,
            self.right,
        ))
    }
}

impl<'data, T, F, R> IndexedParallelIterator for MapWindows<'data, T, F>
where
    T: Sync,
    F: Fn(&'data [T], WindowCenter) -> R + Sync + Send,
    R: Send,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        centers_len(self.boundary, self.slice.len(), self.left, self.right)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        let MapWindows {
            slice,
            left,
            right,
            boundary,
            map_op,
        } = self;
        let start = match boundary {
            // Every element is a center, starting from index 0.
            WindowBoundary::Truncate => 0,
            // The first center with a complete left neighborhood.
            WindowBoundary::Skip => Ord::min(left, slice.len()),
        };
        callback.callback(MapWindowsProducer {
            slice,
            base: 0,
            start,
            len: centers_len(boundary, slice.len(), left, right),
            left,
            right,
            map_op: &map_op,
        })
    }
}

struct MapWindowsProducer<'data, 'f, T: Sync, F> {
    /// A subslice of the original slice, extended on both sides with just
    /// enough "halo" to cover the neighborhoods of every center here.
    slice: &'data [T],
    /// The index of slice[0] in the original slice.
    base: usize,
    /// The index within slice of the first center element.
    start: usize,
    /// The number of center elements handled by this producer.
    len: usize,
    left: usize,
    right: usize,
    map_op: &'f F,
}

impl<'data, 'f, T: Sync, F, R> Producer for MapWindowsProducer<'data, 'f, T, F>
where
    F: Fn(&'data [T], WindowCenter) -> R + Sync,
{
    type Item = R;
    type IntoIter = MapWindowsIter<'data, 'f, T, F>;

    fn into_iter(self) -> Self::IntoIter {
        MapWindowsIter {
            slice: self.slice,
            base: self.base,
            front: self.start,
            back: self.start + self.len,
            left: self.left,
            right: self.right,
            map_op: self.map_op,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index <= self.len);
        // The last center of the left half is start + index - 1, so the
        // left half needs its slice to reach start + index + right.
        let left_end = Ord::min(self.slice.len(), self.start + index + self.right);
        // The first center of the right half is start + index, so the
        // right half may drop everything before its left halo.
        let right_cut = (self.start + index).saturating_sub(self.left);
        (
            MapWindowsProducer {
                slice: &self.slice[..left_end],
                base: self.base,
                start: self.start,
                len: index,
                left: self.left,
                right: self.right,
                map_op: self.map_op,
            },
            MapWindowsProducer {
                slice: &self.slice[right_cut..],
                base: self.base + right_cut,
                start: self.start + index - right_cut,
                len: self.len - index,
                left: self.left,
                right: self.right,
                map_op: self.map_op,
            },
        )
    }
}

struct MapWindowsIter<'data, 'f, T: Sync, F> {
    slice: &'data [T],
    base: usize,
    /// Index within slice of the next center from the front.
    front: usize,
    /// Index within slice one past the last center.
    back: usize,
    left: usize,
    right: usize,
    map_op: &'f F,
}

impl<'data, T: Sync, F, R> MapWindowsIter<'data, '_, T, F>
where
    F: Fn(&'data [T], WindowCenter) -> R,
{
    fn map_at(&self, center: usize) -> R {
        // Truncation only ever happens at the true ends of the original
        // slice, because split_at keeps enough halo on the inner edges.
        let window_start = center.saturating_sub(self.left);
        let window_end = Ord::min(self.slice.len(), center + self.right + 1);
        let window = &self.slice[window_start..window_end];
        let center = WindowCenter {
            index: self.base + center,
            position: center - window_start,
        };
        (self.map_op)(window, center)
    }
}

impl<'data, T: Sync, F, R> Iterator for MapWindowsIter<'data, '_, T, F>
where
    F: Fn(&'data [T], WindowCenter) -> R,
{
    type Item = R;

    fn next(&mut self) -> Option<R> {
        if self.front == self.back {
            return None;
        }
        let center = self.front;
        self.front += 1;
        Some(self.map_at(center))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl<'data, T: Sync, F, R> ExactSizeIterator for MapWindowsIter<'data, '_, T, F>
where
    F: Fn(&'data [T], WindowCenter) -> R,
{
    fn len(&self) -> usize {
        self.back - self.front
    }
}

impl<'data, T: Sync, F, R> DoubleEndedIterator for MapWindowsIter<'data, '_, T, F>
where
    F: Fn(&'data [T], WindowCenter) -> R,
{
    fn next_back(&mut self) -> Option<R> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        Some(self.map_at(self.back))
    }
}
