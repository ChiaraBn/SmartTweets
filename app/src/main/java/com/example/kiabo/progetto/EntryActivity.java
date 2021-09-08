package com.example.kiabo.progetto;

import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.*;
import android.view.*;
import android.widget.*;
import android.content.*;

import com.android.volley.*;
import com.android.volley.toolbox.*;
import com.example.kiabo.myapplication.R;

import org.json.*;
import java.util.*;


/**
 * Attivita' di controllo che permette all'utente di scegliere
 * il museo di appartenenza, su cui basare poi i dataset per la
 * classificazione dei messaggi.
 */
public class EntryActivity extends AppCompatActivity {

    Spinner spinnerMusei;
    Button btnGroup;

    public String museo = "none";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_entry);

        spinnerMusei = (Spinner) findViewById(R.id.spinnerMusei);

        // Contatta il server per avere la lista dei musei
        getMuseums();

        btnGroup = findViewById(R.id.btnGroup);
        btnGroup.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                museo = spinnerMusei.getSelectedItem().toString();
                Log.e("museo: ", museo);

                postMuseum();

                Intent homeIntent = new Intent(EntryActivity.this, HomeActivity.class);
                startActivity(homeIntent);
            }
        });
    }

    /**
     * Contatta il server per ottenere il gruppo a cui appartiene il museo.
     */
    public void postMuseum() {
        String URL = "http://" + HomeActivity.GENERAL_IP + ":5000/groups";
        final RequestQueue queue = Volley.newRequestQueue(EntryActivity.this);

        Map<String, String> postParam = new HashMap<>();
        postParam.put("name", museo);

        JsonObjectRequest obj = new JsonObjectRequest(
                Request.Method.POST,
                URL,
                new JSONObject(postParam),

                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {
                        Log.d("VOLLEY", response.toString());
                        try {
                            String gruppo = response.getString("gruppo");

                            Toast.makeText(EntryActivity.this, "Group: "+gruppo,
                                    Toast.LENGTH_LONG).show();

                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Log.e("VOLLEY", error.toString());
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

        RequestQueue requestQueue = Volley.newRequestQueue(EntryActivity.this);
        requestQueue.add(obj);
    }


    /**
     * Metodo che permette di riempire lo spinner con i nomi
     * dei musei che gli vengono passati dal server.
     */
    public void getMuseums () {
        String URL = "http://" + HomeActivity.GENERAL_IP + ":5000/museums";
        final RequestQueue queue = Volley.newRequestQueue(EntryActivity.this);
        ArrayList<String> musei = new ArrayList<>();

        JsonObjectRequest obj = new JsonObjectRequest(
                Request.Method.GET,
                URL,
                null,

                new Response.Listener<JSONObject>() {
                    @Override
                    public void onResponse(JSONObject response) {

                        try {
                            JSONArray jsonArray = response.getJSONArray("musei");
                            System.out.println ("JSONArray: " + jsonArray);

                            for (int i = 0; i < jsonArray.length(); i++) {
                                musei.add((String) jsonArray.get(i));
                            }

                            Log.e("size: ", String.valueOf(musei.size()));

                            ArrayAdapter<String> spinnerAdapter = new ArrayAdapter<String> (
                                    EntryActivity.this,
                                    android.R.layout.simple_spinner_dropdown_item,
                                    musei
                            );

                            spinnerAdapter.setDropDownViewResource
                                    (android.R.layout.simple_spinner_dropdown_item);
                            spinnerMusei.setAdapter(spinnerAdapter);
                        }
                        catch (JSONException e) {
                            e.printStackTrace();
                        }
                    }
                },

                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Log.e("error: ", error.toString());
                    }
                }
        );

        // Modifico il timeout massimo per raggiungere il server
        obj.setRetryPolicy(new DefaultRetryPolicy(
                50000, DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                DefaultRetryPolicy.DEFAULT_BACKOFF_MULT
        ));

        // Aggiungo la richista
        queue.add(obj);
    }
}
