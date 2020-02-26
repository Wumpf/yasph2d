use crate::units::*;
use cgmath::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;

pub struct ScratchBuffer<T: Copy, TStorage: Copy> {
    pub buffer: Vec<T>,
    store: Rc<RefCell<ScratchBufferTypeStore<TStorage>>>,
}

impl<'a, T: Copy, TStorage: Copy> Drop for ScratchBuffer<T, TStorage> {
    fn drop(&mut self) {
        let mut new_buffer_data_owner = Vec::new();
        std::mem::swap(&mut new_buffer_data_owner, &mut self.buffer);
        self.store.borrow_mut().return_buffer(new_buffer_data_owner);
    }
}

struct ScratchBufferTypeStore<T: Copy> {
    buffers: Vec<Vec<T>>,
}

impl<TStorage: Copy> ScratchBufferTypeStore<TStorage> {
    fn new() -> ScratchBufferTypeStore<TStorage> {
        ScratchBufferTypeStore::<TStorage> { buffers: Vec::new() }
    }

    fn get_buffer<T: Copy>(&mut self, size: usize, default: T) -> Vec<T> {
        assert_eq!(std::mem::size_of::<TStorage>(), std::mem::size_of::<T>());
        assert_eq!(std::mem::align_of::<TStorage>(), std::mem::align_of::<T>());

        match self.buffers.pop() {
            Some(buffer) => {
                let mut buffer = std::mem::ManuallyDrop::new(buffer);
                let mut new_buffer = unsafe { Vec::from_raw_parts(buffer.as_mut_ptr() as *mut T, buffer.len(), buffer.capacity()) };
                new_buffer.resize(size, default);
                new_buffer
            }
            None => vec![default; size],
        }
    }

    fn return_buffer<T: Copy>(&mut self, buffer: Vec<T>) {
        let mut buffer = std::mem::ManuallyDrop::new(buffer);
        let new_buffer = unsafe { Vec::from_raw_parts(buffer.as_mut_ptr() as *mut TStorage, buffer.len(), buffer.capacity()) };
        self.buffers.push(new_buffer);
    }
}

pub struct ScratchBufferStore {
    buffers_real: Rc<RefCell<ScratchBufferTypeStore<Real>>>,
    buffers_vector: Rc<RefCell<ScratchBufferTypeStore<Vector>>>,
}

#[allow(clippy::new_without_default)]
impl ScratchBufferStore {
    pub fn new() -> ScratchBufferStore {
        ScratchBufferStore {
            buffers_real: Rc::new(RefCell::new(ScratchBufferTypeStore::new())),
            buffers_vector: Rc::new(RefCell::new(ScratchBufferTypeStore::new())),
        }
    }

    pub fn get_buffer_real(&self, size: usize) -> ScratchBuffer<Real, Real> {
        ScratchBuffer::<Real, Real> {
            buffer: self.buffers_real.borrow_mut().get_buffer(size, 0.0),
            store: Rc::clone(&self.buffers_real),
        }
    }

    pub fn get_buffer_uint(&self, size: usize) -> ScratchBuffer<u32, Real> {
        ScratchBuffer::<u32, Real> {
            buffer: self.buffers_real.borrow_mut().get_buffer(size, 0),
            store: Rc::clone(&self.buffers_real),
        }
    }

    pub fn get_buffer_vector(&self, size: usize) -> ScratchBuffer<Vector, Vector> {
        ScratchBuffer::<Vector, Vector> {
            buffer: self.buffers_vector.borrow_mut().get_buffer(size, Vector::zero()),
            store: Rc::clone(&self.buffers_vector),
        }
    }

    pub fn get_buffer_point(&self, size: usize) -> ScratchBuffer<Point, Vector> {
        ScratchBuffer::<Point, Vector> {
            buffer: self.buffers_vector.borrow_mut().get_buffer(size, Point::origin()),
            store: Rc::clone(&self.buffers_vector),
        }
    }
}
