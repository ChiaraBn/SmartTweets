
package com.example.kiabo.progetto;

import android.Manifest;
import android.content.*;
import android.graphics.Color;
import android.os.*;
import android.app.Activity;
import android.text.*;
import android.util.Log;
import android.view.View;
import android.widget.*;

import com.android.volley.*;
import com.android.volley.toolbox.*;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.*;

import com.example.kiabo.myapplication.R;
import com.twitter.sdk.android.core.*;
import com.twitter.sdk.android.core.TwitterException;
import com.twitter.sdk.android.core.identity.TwitterLoginButton;
import com.twitter.sdk.android.tweetcomposer.*;


/**
 * Attivita' che controlla il fulcro principale del progetto.
 * Da qui sara' possibile analizzare il proprio messaggio, ricevere l'esito
 * di esso e, con questo, anche eventuali suggerimenti.
 * E' inoltre possibile accedere a Twitter e pubblicare il messaggio
 * dall'applicazione stessa, tramite l'apposito login button.
 */
public class HomeActivity extends Activity {

    Button btnSend ;
    Spinner spinnerImages;
    TextView txtCaratteri, txtSuggestions;
    String msg;
    ImageView imgColore;
    View viewResult ;
    EditText editMsg ;

    TwitterLoginButton loginButton;

    /* Conta il numero di immagini */
    int countMedia = 0 ;

    /* Generico indirizzo ip che rappresenta il 'localhost' */
    public static String GENERAL_IP = "10.0.2.2";

    /* Messaggio che deve inviare */
    public Message message = new Message();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Twitter.initialize(this);
        setContentView(R.layout.activity_home);

        viewResult = (View) findViewById(R.id.viewResult);
        viewResult.setVisibility(View.INVISIBLE);

        imgColore = (ImageView) findViewById(R.id.imgColore);
        imgColore.setVisibility(View.INVISIBLE);

        txtSuggestions = findViewById(R.id.txtSuggestions);
        txtSuggestions.setText("");

        spinnerImages = (Spinner) findViewById(R.id.spinnerImages);
        ArrayList<Integer> img = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            img.add(i);

        ArrayAdapter<Integer> spinnerAdapter = new ArrayAdapter<Integer> (
                HomeActivity.this,
                android.R.layout.simple_spinner_dropdown_item,
                img
        );
        spinnerAdapter.setDropDownViewResource
                (android.R.layout.simple_spinner_dropdown_item);
        spinnerImages.setAdapter(spinnerAdapter);


        /* Crea il bottone e i metodi per gestire il composeTweet */
        loginButton = (TwitterLoginButton) findViewById(R.id.login_button);
        loginButton.setCallback(new Callback<TwitterSession>() {
            @Override
            public void success(Result<TwitterSession> result) {

                TwitterSession session = TwitterCore.getInstance().getSessionManager()
                        .getActiveSession();
                TwitterAuthToken authToken = session.getAuthToken();

                //String token = authToken.token;
                //String secret = authToken.secret;

                composeTweet(session);
            }

            @Override
            public void failure(TwitterException exception) {
                Toast.makeText(HomeActivity.this, "Authentication failed",
                        Toast.LENGTH_LONG).show();
            }
        });

        txtCaratteri = findViewById(R.id.txtCaratteri);
        editMsg = (EditText) findViewById(R.id.editMsg);
        editMsg.setText("");
        editMsg.addTextChangedListener(new TextWatcher() {

            /* Conta i caratteri */
            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                String words = String.valueOf(s.length());
                if (Integer.valueOf(words) > 140) {
                    txtCaratteri.setBackgroundColor(Color.RED);
                    loginButton.setEnabled(false);
                }
                else {
                    txtCaratteri.setBackgroundColor(Color.TRANSPARENT);
                    loginButton.setEnabled(true);
                }

                txtCaratteri.setText(words);
            }

            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        btnSend = (Button) findViewById(R.id.btnSend);
        btnSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                try {
                    countMedia = Integer.parseInt(spinnerImages.getSelectedItem().toString());
                }
                catch (NumberFormatException e) {
                    return ;
                }

                /* Prende le caratteristiche del tweet */
                msg = editMsg.getText().toString();
                message.setFull_text(msg);
                message.setNhash(getNum(msg, "#"));
                message.setNmention(getNum(msg, "@"));
                message.setNimg(countMedia);

                System.out.println("count media: "+ countMedia);

                viewResult.setVisibility(View.VISIBLE);
                sendData();
            }
        });

    }


    /**
     * Crea il compositore di tweet, una volta loggato l'utente.
     * @param session: Sessione utente di Twitter.
     */
    public void composeTweet(TwitterSession session) {

        /* Prende le caratteristiche del tweet */
        msg = editMsg.getText().toString();
        message.setFull_text(msg);
        message.setNhash(getNum(msg, "#"));
        message.setNmention(getNum(msg, "@"));
        message.setNimg(countMedia);

        /* Crea il tweet */
        TweetComposer.Builder builder = new TweetComposer.Builder(this)
                .text(message.getFull_text());
        builder.show();
    }


    /**
     * Metodo che ritorna il numero di hashtag e di menzioni nel tweet.
     * @param msg: Messaggio scritto in input.
     * @param symbol: Simbolo che occorre contare ('@' oppure '#').
     */
    private int getNum(String msg, String symbol) {
        int count = 0;

        String[] text = msg.split(" ");
        for (String word : text) {
            if (word.startsWith(symbol)) {
               count++;
            }
        }
        return count;
    }


    /**
     * Passa il risultato dell'attivita' al pulsante composeTweet.
     */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        loginButton.onActivityResult(requestCode, resultCode, data);
    }


    /**
     * Metodo che permette di inviare il messaggio al server
     * in formato JSON.
     */
    public void sendData() {
        String URL = "http://"+ GENERAL_IP +":5000/api";

        Map<String, Integer> postParam = new HashMap<>();
        postParam.put("NIMG", message.getNimg());
        postParam.put("NHASH", message.getNhash());
        postParam.put("NMENTION", message.getNmention());

        JsonObjectRequest obj = new JsonObjectRequest(
                Request.Method.POST,
                URL,
                new JSONObject(postParam),

                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {
                        Log.d("VOLLEY", response.toString());
                        try {
                            JSONArray suggestion = response.getJSONArray("suggestion");
                            ArrayList <String> labels = new ArrayList<>();

                            if (response.getString("result").equals("Not good")) {
                                imgColore.setVisibility(View.VISIBLE);
                                imgColore.setImageResource(R.drawable.sferarossa);

                                for (int i = 0; i < suggestion.length(); i++) {
                                    labels.add(suggestion.getString(i));
                                }

                                if (labels.size() == 1) {
                                    txtSuggestions.setText("No suggestions available");
                                }
                                else {
                                    txtSuggestions.setText("You could..\n");
                                    for (int i = 0; i < labels.size(); i++)
                                        txtSuggestions.append(labels.get(i) + "\n");
                                }
                            }
                            else {
                                imgColore.setVisibility(View.VISIBLE);
                                imgColore.setImageResource(R.drawable.sferaverde);
                                txtSuggestions.setText("Good Job!");
                            }

                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Log.e("VOLLEY", error.toString());
                        txtSuggestions.setText(error.toString());
                    }
                })

        {
            // Passa alcuni headers
            @Override
            public Map<String, String> getHeaders() throws AuthFailureError {
                HashMap<String, String> headers = new HashMap<String, String>();
                headers.put("Content-Type", "application/json; charset=utf-8");
                return headers;
            }
        };

        // Modifico il timeout massimo per raggiungere il server
        obj.setRetryPolicy(new DefaultRetryPolicy(
                50000, DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
        ));

        RequestQueue requestQueue = Volley.newRequestQueue(HomeActivity.this);
        requestQueue.add(obj);

    }
}
