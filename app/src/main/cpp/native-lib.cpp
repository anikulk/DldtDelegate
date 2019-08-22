#include <jni.h>
#include <string>
#include <fcntl.h>
#include <fstream>
#include <stdio.h>
#include <sys/types.h>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "DLDTDelegate.h"
#include "UseDelegate.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#define TFLITE_DELEGATE "tflite_delegate"
/*
tflite::Interpreter::TfLiteDelegate* CreateNNAPIDelegate() {
  return tflite::NnApiDelegate(),
      // NnApiDelegate() returns a singleton, so provide a no-op deleter.
      [](TfLiteDelegate*) {});
}
*/

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_tflite_1delegate_MainActivity_runInference(
        JNIEnv* env,
        jobject /* this */,
         jobject assetManager) {
    std::string hello = "Hello from C++";
    std::string tflite_path = "terse_rnnt_107M_joint.tflite";
    std::vector<float> test_image = {238,  165,  186,  62,  21,  105,  119,  62,  53,  38,  130,  62,  211,  244,  86,  61,  94,  0,  168,  62,  226,  114,  206,  62,  201,  1,  105,  62,  164,  112,  176,  62,  165,  167,  99,  62,  138,  132,  33,  63,  192,  103,  38,  63,  253,  35,  35,  63,  221,  199,  182,  62,  199,  49,  225,  61,  233,  117,  91,  63,  205,  93,  44,  63,  193,  177,  60,  62,  183,  2,  77,  63,  79,  215,  144,  61,  111,  80,  206,  62,  141,  136,  210,  62,  5,  15,  15,  61,  118,  77,  185,  62,  126,  251,  153,  62,  110,  161,  235,  62,  235,  71,  200,  62,  131,  193,  43,  62,  219,  254,  77,  63,  219,  103,  239,  62,  241,  167,  174,  61,  110,  82,  123,  63,  7,  72,  93,  62,  240,  38,  184,  62,  225,  36,  129,  62,  182,  226,  79,  62,  223,  94,  64,  63,  218,  8,  14,  63,  101,  98,  38,  63,  247,  242,  27,  63,  12,  222,  159,  61,  241,  176,  100,  63,  174,  40,  10,  63,  164,  211,  9,  63,  84,  40,  171,  62,  86,  254,  32,  63,  61,  29,  202,  62,  191,  104,  4,  63,  25,  129,  94,  63,  101,  201,  103,  63,  224,  157,  202,  61,  94,  196,  46,  63,  173,  202,  178,  62,  238,  35,  34,  63,  237,  34,  94,  61,  141,  40,  153,  62,  50,  99,  74,  62,  57,  67,  94,  62,  198,  228,  45,  62,  235,  118,  28,  63,  151,  125,  15,  63,  225,  184,  236,  62,  68,  9,  85,  63,  240,  165,  131,  62,  231,  206,  222,  61,  61,  222,  15,  60,  237,  213,  119,  63,  124,  79,  83,  63,  133,  59,  160,  62,  107,  242,  109,  63,  245,  194,  61,  63,  22,  158,  120,  62,  17,  182,  112,  63,  197,  40,  11,  63,  123,  40,  79,  63,  127,  95,  142,  62,  112,  83,  123,  63,  82,  40,  55,  62,  241,  195,  49,  63,  17,  33,  48,  63,  66,  188,  81,  60,  79,  222,  19,  61,  154,  250,  88,  62,  175,  102,  88,  63,  223,  87,  75,  63,  55,  148,  178,  61,  90,  253,  45,  63,  36,  190,  74,  62,  128,  233,  33,  62,  140,  21,  181,  62,  22,  212,  35,  63,  226,  192,  41,  62,  58,  83,  159,  62,  82,  236,  118,  63,  232,  194,  34,  63,  210,  70,  40,  62,  212,  13,  28,  63,  137,  68,  65,  62,  103,  28,  240,  62,  219,  151,  28,  62,  18,  82,  61,  61,  68,  118,  250,  62,  201,  184,  11,  63,  4,  95,  108,  63,  152,  90,  232,  61,  198,  21,  84,  63,  0,  248,  199,  62,  153,  154,  207,  62,  48,  125,  53,  63,  113,  243,  100,  63,  65,  140,  72,  63,  4,  191,  77,  63,  14,  220,  63,  62,  43,  92,  196,  62,  217,  69,  174,  62,  91,  210,  158,  61,  35,  232,  116,  63,  184,  230,  99,  63,  42,  226,  240,  62,  89,  124,  213,  60,  56,  194,  74,  63,  68,  165,  223,  62,  89,  105,  56,  62,  74,  174,  242,  62,  251,  84,  226,  61,  236,  146,  127,  63,  174,  170,  86,  62,  164,  95,  143,  61,  23,  213,  18,  63,  111,  102,  117,  63,  127,  210,  76,  63,  130,  32,  37,  61,  15,  202,  126,  62,  12,  99,  13,  63,  157,  53,  186,  62,  201,  129,  110,  63,  9,  174,  93,  61,  146,  138,  191,  62,  2,  244,  140,  62,  140,  173,  237,  62,  208,  148,  106,  63,  10,  174,  78,  63,  58,  233,  128,  62,  236,  95,  96,  63,  47,  80,  176,  62,  195,  215,  234,  62,  217,  120,  197,  62,  100,  179,  160,  62,  47,  39,  81,  63,  43,  154,  50,  63,  195,  159,  246,  62};

    float out[10];
     AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
     AAsset *asset = AAssetManager_open(mgr, tflite_path.c_str(), O_RDONLY);
     off_t start = 0;
     off_t length = AAsset_getLength(asset);
     int mmap_fd_ = AAsset_openFileDescriptor(asset, &start, &length);
     if (mmap_fd_ < 0) {
         __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "Error", 1);
         AAsset_close(asset);
     }
    off_t  dataSize = AAsset_getLength(asset);
    const void* const memory = AAsset_getBuffer(asset);

    // Use as const char*
    const char* const memChar = (const char*) memory;

    // Create a new Buffer for the FlatBuffer with the size needed.
    // It has to exist alongside the FlatBuffer model as long as the model shall exist!
    char* flatBuffersBuffer; //(declared in the header file of the class in which I use this).
    flatBuffersBuffer = new char[dataSize];
    __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "Copying assets buffer to flatbuffers buffer, size: %d", dataSize, 1);

    for(int i = 0; i < dataSize; i++)
    {
        flatBuffersBuffer[i] = memChar[i];
    }

    auto model = tflite::FlatBufferModel::BuildFromBuffer(flatBuffersBuffer, dataSize);
    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, op_resolver)(&interpreter);
    std::vector <float>* input0 = interpreter->typed_input_tensor<std::vector <float>>(0);
    std::vector <float>* joint = interpreter->typed_input_tensor<std::vector<float>>(1);

    if(interpreter->AllocateTensors() != kTfLiteOk){
        __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "Failed to allocate tensors\n", 1);
        exit(0);
     }

    input0 = &test_image;
    joint = &test_image;
    bool use_nnapi = false;

    TfLiteDelegate* dldt_tf_delegate = CreateDLDTDelegate();
    if(use_nnapi) {
        if (interpreter->ModifyGraphWithDelegate(tflite::NnApiDelegate()) != kTfLiteOk) {
              __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "Failed to apply nnapi delegate.", 1);
        } else {
              __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "Applied nnapi delegate.", 1);
        }
    }
    else {
        if (interpreter->ModifyGraphWithDelegate(dldt_tf_delegate) != kTfLiteOk) {
            __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "Failed to apply dldt delegate.", 1);
        } else {
            __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "Applied dldt delegate.", 1);
                }
    }



    if(interpreter->Invoke() != kTfLiteOk) {
        __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE,"Failed to invoke", 1);
    }
    else {
        __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "Invoked llrtdelegate.", 1);
    }

    float* output = interpreter->typed_output_tensor<float>(0);

    for (int i = 0; i < 10; i++) {
        __android_log_print(ANDROID_LOG_VERBOSE, TFLITE_DELEGATE, "%d = %f\n", i, *(output + i), 1);
    }
    return env->NewStringUTF(hello.c_str());
}
