
package com.example.kiabo.progetto;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.widget.Toast;

import com.twitter.sdk.android.tweetcomposer.TweetUploadService;

/**
 * Dopo aver tentato di mandare un tweet, il TweetUploadService manda
 * un Intent con il risulato dell'azione.
 * Questa classe ha lo scopo di ricevere l'Intent.
 */
public class MyResultReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        if (TweetUploadService.UPLOAD_SUCCESS.equals(intent.getAction())) {
            // Successo
            Toast.makeText(context, "Tweet SENT!",
                    Toast.LENGTH_LONG).show();
        }
        else if (TweetUploadService.UPLOAD_FAILURE.equals(intent.getAction())) {
            // Fallimento
            Toast.makeText(context, "ERROR",
                    Toast.LENGTH_LONG).show();
        }
        else if (TweetUploadService.TWEET_COMPOSE_CANCEL.equals(intent.getAction())) {
            // Annulla
            Toast.makeText(context, "Tweet CANCELLED",
                    Toast.LENGTH_LONG).show();
        }
    }
}