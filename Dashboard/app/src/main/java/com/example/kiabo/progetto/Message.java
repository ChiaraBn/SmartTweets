
package com.example.kiabo.progetto;


/**
 * Classe che implementa l'oggetto 'messaggio' che deve essere inviato.
 */
public class Message {
    private String full_text = "";
    private int nhash = 0;
    private int nmention = 0;
    private int nimg = 0;

    /**
     * Metodo costruttore per la classe.
     * E' un metodo che inizializza gli attributi con
     * valori di default.
     */
    public Message () {
        this.full_text = "none";
        this.nhash = 0;
        this.nimg = 0;
        this.nmention = 0;
    }

    /**
     * Metodo costruttore che inizializza gli attributi
     * con valori passati in input.
     * @param text: Testo del messaggio.
     * @param img: Numero di immagini.
     * @param hash: Numero di hashtag.
     * @param mention: Numero di menzioni.
     */
    public Message (String text, int img, int hash, int mention) {
        this.full_text = text;
        this.nimg = img;
        this.nhash = hash;
        this.nmention = mention;
    }


    // Metodi SET e GET per gli attributi di classe
    public String getFull_text() {
        return full_text;
    }

    public void setFull_text(String full_text) {
        this.full_text = full_text;
    }

    public int getNhash() {
        return nhash;
    }

    public void setNhash(int nhash) {
        this.nhash = nhash;
    }

    public int getNmention() {
        return nmention;
    }

    public void setNmention(int nmention) {
        this.nmention = nmention;
    }

    public int getNimg() {
        return nimg;
    }

    public void setNimg(int nimg) {
        this.nimg = nimg;
    }
}
