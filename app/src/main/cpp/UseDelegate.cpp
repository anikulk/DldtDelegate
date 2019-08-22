//
// Created by anishak on 8/19/19.
//
#include <stdio.h>
#include <vector>
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
#include "TfLiteArrayView.h"
#include <iostream>

#include "DLDTDelegate.h"

TfLiteRegistration GetMyDelegateNodeRegistration() {
  // This is the registration for the Delegate Node that gets added to
  // the TFLite graph instead of the subGraph it replaces.
  // It is treated as a an OP node. But in our case
  // Init will initialize the delegate
  // Invoke will run the delegate graph.
  // Prepare for preparing the delegate.
      // Free for any cleaning needed by the delegate.
      TfLiteRegistration kernel_registration;
      kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
      kernel_registration.custom_name = "DLDTDelegate";
      kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
        delete reinterpret_cast<DLDTDelegate*>(buffer);
      };
      kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                       size_t) -> void* {
        // In the node init phase, initialize MyDelegate instance
        const TfLiteDelegateParams* delegate_params =
            reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        DLDTDelegate* dldt_delegate = new DLDTDelegate;
        if (!dldt_delegate->Init(context, delegate_params)) {
          return nullptr;
        }
        return dldt_delegate;
      };
    kernel_registration.invoke = [](TfLiteContext* context,
                                       TfLiteNode* node) -> TfLiteStatus {
        DLDTDelegate* kernel = reinterpret_cast<DLDTDelegate*>(node->user_data);
        return kernel->Invoke(context, node);
    };
    kernel_registration.prepare = [](TfLiteContext* context,
                                        TfLiteNode* node) -> TfLiteStatus {
        DLDTDelegate* kernel = reinterpret_cast<DLDTDelegate*>(node->user_data);
        return kernel->Prepare(context, node);
    };

    return kernel_registration;
}

// TfLiteDelegate methods

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
    // Claim all nodes that can be evaluated by the delegate and ask the
    // framework to update the graph with delegate kernel instead.
    // Reserve 1 element, since we need first element to be size.d
    __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "In UseDelegate DelegatePrepare.", 1);
    std::vector<int> supported_nodes(1);
    TfLiteIntArray* plan;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));
    TfLiteNode* node;
    TfLiteRegistration* registration;
    //array_view = new TfLiteIntArrayView(plan);
    for (auto node_index : TfLiteIntArrayView(plan)) {
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (DLDTDelegate::SupportedOp(registration)) {
      supported_nodes.push_back(node_index);
    }
    }
    // Set first element to the number of nodes to replace.
    supported_nodes[0] = supported_nodes.size() - 1;
    TfLiteRegistration dldt_delegate_kernel_registration =
      GetMyDelegateNodeRegistration();

    // This call split the graphs into subgraphs, for subgraphs that can be
    // handled by the delegate, it will replace it with a
    // 'my_delegate_kernel_registration'
    return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, dldt_delegate_kernel_registration,
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()), delegate);
}

void FreeBufferHandle(TfLiteContext* context, TfLiteDelegate* delegate,
                      TfLiteBufferHandle* handle) {
  // Do any cleanups.
}

TfLiteStatus CopyToBufferHandle(TfLiteContext* context,
                                TfLiteDelegate* delegate,
                                TfLiteBufferHandle buffer_handle,
                                TfLiteTensor* tensor) {
  // Copies data from tensor to delegate buffer if needed.
  return kTfLiteOk;
}

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle,
                                  TfLiteTensor* tensor) {
  // Copies the data from delegate buffer into the tensor raw memory.
  return kTfLiteOk;
}

TfLiteDelegate* CreateDLDTDelegate() {
    TfLiteDelegate* delegate = new TfLiteDelegate;
    __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "In UseDelegate CreateDLDTDdelegate.", 1);
    std::cout << "In UseDelegate CreateDelegate\n";

    delegate->data_ = nullptr;
    delegate->flags = kTfLiteDelegateFlagsNone;
    delegate->Prepare = &DelegatePrepare;
    // This cannot be null.
    delegate->CopyFromBufferHandle = &CopyFromBufferHandle;
    // This can be null.
    delegate->CopyToBufferHandle = &CopyToBufferHandle;
    // This can be null.
    delegate->FreeBufferHandle = &FreeBufferHandle;

    return delegate;
}
