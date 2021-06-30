// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::{
    array::{Array, PrimitiveArray},
    bitmap::{Bitmap, MutableBitmap},
    buffer::{Buffer, MutableBuffer},
    error::{ArrowError, Result},
    types::NativeType,
};

use super::maybe_usize;
use super::Index;

// take implementation when neither values nor indices contain nulls
fn take_no_validity<T: NativeType, I: Index, F: Fn(&[T], usize) -> Result<T>>(
    values: &[T],
    indices: &[I],
    get: F,
) -> Result<(Buffer<T>, Option<Bitmap>)> {
    let values = indices.iter().map(|index| {
        let index = maybe_usize::<I>(*index)?;
        (get)(values, index)
    });
    let buffer = MutableBuffer::try_from_trusted_len_iter(values)?;

    Ok((buffer.into(), None))
}

// take implementation when only values contain nulls
fn take_values_validity<
    T: NativeType,
    I: Index,
    F: Fn(&[T], &Bitmap, usize) -> Result<(T, bool)>,
>(
    values: &PrimitiveArray<T>,
    indices: &[I],
    get: F,
) -> Result<(Buffer<T>, Option<Bitmap>)> {
    let mut validity = MutableBitmap::with_capacity(indices.len());

    let validity_values = values.validity().as_ref().unwrap();

    let values_values = values.values();

    let values = indices.iter().map(|index| {
        let index = maybe_usize::<I>(*index)?;

        let (value, is_valid) = (get)(values_values, validity_values, index)?;
        if is_valid {
            validity.push(true);
        } else {
            validity.push(false);
        }
        Result::Ok(value)
    });
    let buffer = MutableBuffer::try_from_trusted_len_iter(values)?;

    Ok((buffer.into(), validity.into()))
}

// take implementation when only indices contain nulls
fn take_indices_validity<T: NativeType, I: Index, F: Fn(&[T], usize) -> Result<T>>(
    values: &[T],
    indices: &PrimitiveArray<I>,
    get: F,
) -> Result<(Buffer<T>, Option<Bitmap>)> {
    let values = indices.iter().map(|index| match index {
        Some(index) => {
            let index = maybe_usize::<I>(*index)?;
            (get)(values, index)
        }
        None => Ok(T::default()),
    });

    let buffer = MutableBuffer::try_from_trusted_len_iter(values)?;

    Ok((buffer.into(), indices.validity().clone()))
}

// take implementation when both values and indices contain nulls
fn take_values_indices_validity<
    T: NativeType,
    I: Index,
    F: Fn(&[T], &Bitmap, usize) -> Result<(T, bool)>,
>(
    values: &PrimitiveArray<T>,
    indices: &PrimitiveArray<I>,
    get: F,
) -> Result<(Buffer<T>, Option<Bitmap>)> {
    let mut validity = MutableBitmap::with_capacity(indices.len());

    let values_validity = values.validity().as_ref().unwrap();

    let values_values = values.values();
    let values = indices.iter().map(|index| match index {
        Some(index) => {
            let index = maybe_usize::<I>(*index)?;
            let (value, is_valid) = get(values_values, values_validity, index)?;
            validity.push(is_valid);
            Result::Ok(value)
        }
        None => {
            validity.push(false);
            Ok(T::default())
        }
    });
    let buffer = MutableBuffer::try_from_trusted_len_iter(values)?;
    Ok((buffer.into(), validity.into()))
}

/// `take` implementation for primitive arrays
pub fn take<T: NativeType, I: Index>(
    values: &PrimitiveArray<T>,
    indices: &PrimitiveArray<I>,
) -> Result<PrimitiveArray<T>> {
    let indices_has_validity = indices.null_count() > 0;
    let values_has_validity = values.null_count() > 0;
    let (buffer, validity) = match (values_has_validity, indices_has_validity) {
        (false, false) => take_no_validity(values.values(), indices.values(), |values, index| {
            values
                .get(index)
                .copied()
                .ok_or(ArrowError::KeyOverflowError)
        })?,
        (true, false) => {
            take_values_validity(values, indices.values(), |values, validity, index| {
                Ok((
                    values
                        .get(index)
                        .copied()
                        .ok_or(ArrowError::KeyOverflowError)?,
                    validity.get(index).ok_or(ArrowError::KeyOverflowError)?,
                ))
            })?
        }
        (false, true) => take_indices_validity(values.values(), indices, |values, index| {
            values
                .get(index)
                .copied()
                .ok_or(ArrowError::KeyOverflowError)
        })?,
        (true, true) => {
            take_values_indices_validity(values, indices, |values, validity, index| {
                Ok((
                    values
                        .get(index)
                        .copied()
                        .ok_or(ArrowError::KeyOverflowError)?,
                    validity.get(index).ok_or(ArrowError::KeyOverflowError)?,
                ))
            })?
        }
    };

    Ok(PrimitiveArray::<T>::from_data(
        values.data_type().clone(),
        buffer,
        validity,
    ))
}

/// `take` implementation for primitive arrays
/// # Safety
/// The caller must ensure that all indices are in-bounds
pub unsafe fn take_unchecked<T: NativeType, I: Index>(
    values: &PrimitiveArray<T>,
    indices: &PrimitiveArray<I>,
) -> Result<PrimitiveArray<T>> {
    let indices_has_validity = indices.null_count() > 0;
    let values_has_validity = values.null_count() > 0;
    let (buffer, validity) = match (values_has_validity, indices_has_validity) {
        (false, false) => take_no_validity(values.values(), indices.values(), |values, index| {
            Ok(*values.get_unchecked(index))
        })?,
        (true, false) => {
            take_values_validity(values, indices.values(), |values, validity, index| {
                Ok((*values.get_unchecked(index), validity.get_unchecked(index)))
            })?
        }
        (false, true) => take_indices_validity(values.values(), indices, |values, index| {
            Ok(*values.get_unchecked(index))
        })?,
        (true, true) => {
            take_values_indices_validity(values, indices, |values, validity, index| {
                Ok((*values.get_unchecked(index), validity.get_unchecked(index)))
            })?
        }
    };

    Ok(PrimitiveArray::<T>::from_data(
        values.data_type().clone(),
        buffer,
        validity,
    ))
}
