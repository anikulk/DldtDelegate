#include "tensorflow/lite/context.h"
#include "tensorflow/lite/builtin_ops.h"

#include <android/log.h>

#define TFLITE_DELEGATE "tflite_delegate"

class DLDTDelegate {

public:
    static bool SupportedOp(const TfLiteRegistration* registration) {
        switch(registration->builtin_code) {
        case kTfLiteBuiltinConv2d:
        case kTfLiteBuiltinMean:
        case kTfLiteBuiltinFullyConnected:
            return true;
        default:
            return false;
        }
    }

    TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params);

    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode *node);

    TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode *node);

};