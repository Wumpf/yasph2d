use crate::units::*;
use cgmath::prelude::*;
use std::cell::RefCell;

pub struct ScratchBuffer<'a, T: Copy> {
    pub buffer: Vec<T>,
    store: &'a RefCell<ScratchBufferTypeStore<T>>,
}

impl<'a, T: Copy> Drop for ScratchBuffer<'a, T> {
    fn drop(&mut self) {
        let mut new_buffer_data_owner = Vec::new();
        std::mem::swap(&mut new_buffer_data_owner, &mut self.buffer);
        self.store.borrow_mut().return_buffer(new_buffer_data_owner);
    }
}

struct ScratchBufferTypeStore<T: Copy> {
    buffers: Vec<Vec<T>>,
}

impl<T: Copy> ScratchBufferTypeStore<T> {
    fn new() -> ScratchBufferTypeStore<T> {
        ScratchBufferTypeStore::<T> { buffers: Vec::new() }
    }

    fn get_buffer(&mut self, size: usize, default: T) -> Vec<T> {
        match self.buffers.pop() {
            Some(mut buffer) => {
                buffer.resize(size, default);
                buffer
            }
            None => vec![default; size],
        }
    }

    fn return_buffer(&mut self, buffer: Vec<T>) {
        self.buffers.push(buffer);
    }
}

pub struct ScratchBufferStore {
    buffers_real: RefCell<ScratchBufferTypeStore<Real>>,
    buffers_vector: RefCell<ScratchBufferTypeStore<Vector>>,
    buffers_point: RefCell<ScratchBufferTypeStore<Point>>, // todo, do some unsafe code to fuse this with Vector store
}

impl ScratchBufferStore {
    pub fn new() -> ScratchBufferStore {
        ScratchBufferStore {
            buffers_real: RefCell::new(ScratchBufferTypeStore::new()),
            buffers_vector: RefCell::new(ScratchBufferTypeStore::new()),
            buffers_point: RefCell::new(ScratchBufferTypeStore::new()),
        }
    }

    pub fn get_buffer_real(&mut self, size: usize) -> ScratchBuffer<Real> {
        ScratchBuffer::<Real> {
            buffer: self.buffers_real.borrow_mut().get_buffer(size, 0.0),
            store: &self.buffers_real,
        }
    }

    pub fn get_buffer_vector(&mut self, size: usize) -> ScratchBuffer<Vector> {
        ScratchBuffer::<Vector> {
            buffer: self.buffers_vector.borrow_mut().get_buffer(size, Vector::zero()),
            store: &self.buffers_vector,
        }
    }

    pub fn get_buffer_point(&mut self, size: usize) -> ScratchBuffer<Point> {
        ScratchBuffer::<Point> {
            buffer: self.buffers_point.borrow_mut().get_buffer(size, Point::origin()),
            store: &self.buffers_point,
        }
    }
}
