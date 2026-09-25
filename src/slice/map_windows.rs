use crate::iter::plumbing::*;
use crate::iter::*;

use std::fmt::{self, Debug};

/// Boundary policy for [`ParallelSlice::par_map_windows`].
///
/// [`ParallelSlice::par_map_windows`]: crate::slice::ParallelSlice::par_map_windows
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WindowBoundary {
    /// Windows near the edges of the slice are truncated to the slice bounds,
    /// so every element acts as a center and produces exactly one output.
    Truncate,
    /// Centers whose full `left + 1 + right` window would extend past either
    /// edge of the slice are skipped, so every window seen by the closure is
    /// complete. Only `len - left - right` outputs are produced (or none at
    /// all if the neighborhood is wider than the slice).
    Skip,
}

/// `MapWindows` is an iterator that maps a closure over the neighborhood
/// window of each element of a slice.
///
/// This struct is created by the [`par_map_windows()`] method on
/// [`ParallelSlice`].
///
/// [`par_map_windows()`]: crate::slice::ParallelSlice::par_map_windows
/// [`ParallelSlice`]: crate::slice::ParallelSlice
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct MapWindows<'data, T, F> {
    slice: &'data [T],
    left: usize,
    right: usize,
    boundary: WindowBoundary,
    map: F,
}

impl<T, F: Clone> Clone for MapWindows<'_, T, F> {
    fn clone(&self) -> Self {
        MapWindows {
            slice: self.slice,
            left: self.left,
            right: self.right,
            boundary: self.boundary,
            map: self.map.clone(),
        }
    }
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
        map: F,
    ) -> Self {
        assert!(
            left.checked_add(right).is_some(),
            "par_map_windows: `left + right` overflows `usize`"
        );
        MapWindows {
            slice,
            left,
            right,
            boundary,
            map,
        }
    }
}

impl<'data, T, F, R> ParallelIterator for MapWindows<'data, T, F>
where
    T: Sync,
    F: Fn(&[T], usize, usize) -> R + Sync + Send,
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
        Some(self.len())
    }
}

impl<T, F, R> IndexedParallelIterator for MapWindows<'_, T, F>
where
    T: Sync,
    F: Fn(&[T], usize, usize) -> R + Sync + Send,
    R: Send,
{
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    fn len(&self) -> usize {
        match self.boundary {
            WindowBoundary::Truncate => self.slice.len(),
            WindowBoundary::Skip => self.slice.len().saturating_sub(self.left + self.right),
        }
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        let len = self.len();
        // With `Skip`, the first center is `left` elements into the
        // slice, both locally and in absolute slice coordinates.
        let first_center = match self.boundary {
            WindowBoundary::Truncate => 0,
            WindowBoundary::Skip if len > 0 => self.left,
            WindowBoundary::Skip => 0,
        };
        callback.callback(MapWindowsProducer {
            slice: self.slice,
            first_center,
            abs_center: first_center,
            len,
            left: self.left,
            right: self.right,
            boundary: self.boundary,
            map: &self.map,
        })
    }
}

/// A producer over a subslice of the original slice. The subslice always
/// carries enough "halo" elements on both sides to form the full window of
/// every center it produces, so splitting never needs to copy anything.
struct MapWindowsProducer<'f, 'data, T, F> {
    /// Subslice of the original slice, including halo elements.
    slice: &'data [T],
    /// Index of the first center within `slice`.
    first_center: usize,
    /// Absolute index of the first center within the original slice.
    abs_center: usize,
    /// Number of centers (outputs) produced.
    len: usize,
    left: usize,
    right: usize,
    boundary: WindowBoundary,
    map: &'f F,
}

impl<'f, 'data, T, F, R> Producer for MapWindowsProducer<'f, 'data, T, F>
where
    T: Sync,
    F: Fn(&[T], usize, usize) -> R + Sync,
    R: Send,
{
    type Item = R;
    type IntoIter = MapWindowsIter<'f, 'data, T, F>;

    fn into_iter(self) -> Self::IntoIter {
        MapWindowsIter {
            slice: self.slice,
            first_center: self.first_center,
            abs_center: self.abs_center,
            len: self.len,
            next: 0,
            left: self.left,
            right: self.right,
            boundary: self.boundary,
            map: self.map,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        // The left half keeps its centers `first_center..first_center + index`
        // and retains up to `right` halo elements after the last center.
        let left_end = (self.first_center + index)
            .saturating_add(self.right)
            .min(self.slice.len());
        // The right half starts its centers at `first_center + index` and
        // retains up to `left` halo elements before its first center.
        let right_start = (self.first_center + index).saturating_sub(self.left);
        (
            MapWindowsProducer {
                slice: &self.slice[..left_end],
                first_center: self.first_center,
                abs_center: self.abs_center,
                len: index,
                left: self.left,
                right: self.right,
                boundary: self.boundary,
                map: self.map,
            },
            MapWindowsProducer {
                slice: &self.slice[right_start..],
                first_center: self.first_center + index - right_start,
                abs_center: self.abs_center + index,
                len: self.len - index,
                left: self.left,
                right: self.right,
                boundary: self.boundary,
                map: self.map,
            },
        )
    }
}

/// Sequential iterator over the windows of a single producer. It never
/// clones elements and never allocates: each window is a subslice of the
/// producer's slice, formed in O(1).
struct MapWindowsIter<'f, 'data, T, F> {
    slice: &'data [T],
    first_center: usize,
    abs_center: usize,
    len: usize,
    next: usize,
    left: usize,
    right: usize,
    boundary: WindowBoundary,
    map: &'f F,
}

impl<T, F, R> MapWindowsIter<'_, '_, T, F>
where
    F: Fn(&[T], usize, usize) -> R,
{
    /// Maps the center at relative offset `index` to one output.
    fn map_at(&self, index: usize) -> R {
        let center = self.first_center + index;
        let (start, end) = match self.boundary {
            WindowBoundary::Truncate => (
                center.saturating_sub(self.left),
                center
                    .saturating_add(self.right)
                    .saturating_add(1)
                    .min(self.slice.len()),
            ),
            WindowBoundary::Skip => {
                // The producer's halo guarantees the full window is present.
                (center - self.left, center + self.right + 1)
            }
        };
        let window = &self.slice[start..end];
        (self.map)(window, center - start, self.abs_center + index)
    }
}

impl<T, F, R> Iterator for MapWindowsIter<'_, '_, T, F>
where
    F: Fn(&[T], usize, usize) -> R,
{
    type Item = R;

    fn next(&mut self) -> Option<R> {
        if self.next < self.len {
            let item = self.map_at(self.next);
            self.next += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len - self.next;
        (len, Some(len))
    }
}

impl<T, F, R> DoubleEndedIterator for MapWindowsIter<'_, '_, T, F>
where
    F: Fn(&[T], usize, usize) -> R,
{
    fn next_back(&mut self) -> Option<R> {
        if self.next < self.len {
            self.len -= 1;
            Some(self.map_at(self.len))
        } else {
            None
        }
    }
}

impl<T, F, R> ExactSizeIterator for MapWindowsIter<'_, '_, T, F> where F: Fn(&[T], usize, usize) -> R
{}
