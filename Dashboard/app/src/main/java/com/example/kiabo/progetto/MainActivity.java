
package com.example.kiabo.progetto;

import android.content.Intent;
import android.os.*;
import android.support.v7.app.AppCompatActivity;

import com.example.kiabo.myapplication.R;


/**
 * File principale che avvia l'applicazione.
 * In questa attivita' verra' mostrata una pagina iniziale
 * con il logo ideato per il programma, che restera' attiva per 4 secondi,
 * per poi passare il controllo alla pagina in cui decidere
 * il museo di appartenenza.
 */
public class MainActivity extends AppCompatActivity {

    private static int SPLASH_TIME_OUT = 4000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {;

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent homeIntent = new Intent(MainActivity.this, EntryActivity.class);
                startActivity(homeIntent);
                finish();
            }
        }, SPLASH_TIME_OUT );
    }
}






