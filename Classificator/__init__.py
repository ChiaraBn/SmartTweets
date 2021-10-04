
"""
    File obbligatorio per le directories Python,
    in quanto permette di inizializzare oggetti e ambienti.
    Nello specifico viene inizializzata l'applicazione 'dashboard'.
"""

from flask import Flask

app = Flask(__name__)

from dashboard import routes