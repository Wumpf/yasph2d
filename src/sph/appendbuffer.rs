use std::alloc::{self, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

// Simple thread safe append buffer
// Fails if appended beyond capacity
// (similar to the AppendBuffer concept in shader languages)
pub struct AppendBuffer<T: Copy> {
    capacity: usize,
    size: AtomicUsize,
    data: *mut T,
}

impl<T: Copy> AppendBuffer<T> {
    fn buffer_layout(capacity: usize) -> Layout {
        Layout::from_size_align(std::mem::size_of::<T>() * capacity, std::mem::align_of::<T>()).unwrap()
    }

    pub fn new() -> AppendBuffer<T> {
        AppendBuffer {
            capacity: 0,
            size: AtomicUsize::new(0),
            data: std::ptr::null_mut(),
        }
    }

    // pub fn with_capacity(capacity: usize) -> AppendBuffer<T> {
    //     let allocation = unsafe { alloc::alloc(Self::buffer_layout(capacity)) };
    //     AppendBuffer {
    //         capacity,
    //         size: AtomicUsize::new(0),
    //         data: allocation.cast(),
    //     }
    // }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data, self.size.load(Ordering::Relaxed)) }
    }

    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    // Threadsafe growing!
    pub fn extend_from_slice(&self, slice: &[T]) -> Result<usize, usize> {
        let previous_size = self.size.fetch_add(slice.len(), Ordering::Relaxed);

        if previous_size + slice.len() > self.capacity {
            self.size.fetch_sub(slice.len(), Ordering::Relaxed);
            return Err(previous_size);
        }

        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), self.data.add(previous_size), slice.len());
        }

        Ok(previous_size)
    }

    pub fn resize(&mut self, capacity: usize) {
        if self.capacity >= capacity {
            return;
        }
        self.capacity = capacity;
        unsafe {
            alloc::dealloc(self.data.cast(), Self::buffer_layout(self.capacity));
            self.data = alloc::alloc(Self::buffer_layout(capacity)).cast();
        }
    }

    pub fn clear(&mut self) {
        self.size.store(0, Ordering::Relaxed);
    }
}

impl<T: Copy> Drop for AppendBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.data.cast(), Self::buffer_layout(self.capacity));
        }
    }
}

unsafe impl<T: Copy> Sync for AppendBuffer<T> {}
unsafe impl<T: Copy> Send for AppendBuffer<T> {}
