//
// Created by anishak on 8/19/19.
//
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/nnapi/nnapi_implementation.h"
#include "tensorflow/lite/util.h"

#include <android/log.h>

#define TFLITE_DELEGATE "tflite_delegate"

#ifndef TFLITE_DELEGATE_USEDELEGATE_H
#define TFLITE_DELEGATE_USEDELEGATE_H

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate *delegate);


void FreeBufferHandle(TfLiteContext* context, TfLiteDelegate* delegate,
                      TfLiteBufferHandle* handle);

TfLiteStatus CopyToBufferHandle(TfLiteContext* context,
                            TfLiteDelegate* delegate,
                            TfLiteBufferHandle buffer_handle,
                            TfLiteTensor* tensor);

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                              TfLiteDelegate* delegate,
                              TfLiteBufferHandle buffer_handle,
                              TfLiteTensor* tensor);

// Caller takes ownership of the returned pointer.
TfLiteDelegate* CreateDLDTDelegate();

#endif //TFLITE_DELEGATE_USEDELEGATE_H
