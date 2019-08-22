#include <stdio.h>
#include <vector>
#include <iostream>
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/builtin_ops.h"
#include "DLDTDelegate.h"


TfLiteStatus DLDTDelegate::Init(TfLiteContext* context, const TfLiteDelegateParams* params) {
    __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "In Init.", 1);
    std::cout << "In UseDelegate CreateDelegate\n";
    return kTfLiteOk;
}

TfLiteStatus DLDTDelegate::Prepare(TfLiteContext* context, TfLiteNode *node){
    __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "In Prepare.", 1);
    std::cout << "In UseDelegate CreateDelegate\n";
    return kTfLiteOk;
}

TfLiteStatus DLDTDelegate::Invoke(TfLiteContext* context, TfLiteNode *node){
    __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "In Invoke.", 1);
     std::cout << "In UseDelegate CreateDelegate\n";
    return kTfLiteOk;
}


