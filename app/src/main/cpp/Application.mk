APP_ABI := x86_64
# Android NDK: android-8 is unsupported. Using minimum supported version android-16.
APP_PLATFORM := android-16
# Android NDK: Invalid NDK_TOOLCHAIN_VERSION value: 4.9. GCC is no longer supported. See https://android.googlesource.com/platform/ndk/+/master/docs/ClangMigration.md.
# Android NDK: APP_STL gnustl_static is no longer supported. Please switch to either c++_static or c++_shared. See https://developer.android.com/ndk/guides/cpp-support.html for more information.