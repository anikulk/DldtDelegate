package com.example.tflite_delegate;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;
import android.content.res.AssetManager;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import java.io.*;

import android.content.Context;
public class MainActivity extends AppCompatActivity {
    private final String TAG = "MainActivity";
    private AssetManager mgr;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        FileOutputStream outputStream;

        // Example of a call to a native method
        TextView tv = findViewById(R.id.sample_text);
        mgr = getResources().getAssets();
        tv.setText(runInference(mgr));

    }


    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    private static native String runInference(AssetManager mgr);

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
        }
}
