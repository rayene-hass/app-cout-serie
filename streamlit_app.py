import streamlit as st
import pandas as pd
import math
import io
import json
import gspread
from google.oauth2.service_account import Credentials
import json

def sauvegarder_parametres_gsheet():

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
    worksheet = sheet.worksheet(st.secrets["sheets"]["worksheet_reglages"])

    df = st.session_state.df_nomenclature
    comp_params = st.session_state.get("comp_params", {})

    # Création des lignes à sauvegarder
    sauvegarde = []
    for _, row in df.iterrows():
        comp_key = f"{row.get('Ensemble','')}/{row.get('Sous-Ensemble','')}/{row.get('Composant','')}/{row.get('Fournisseur','')}".strip().lower()
        params = comp_params.get(comp_key, {})
        sauvegarde.append({
            "comp_key": comp_key,
            "law": row.get("Loi spécifique", "Global"),
            "prix_matiere": row.get("Prix matière (€/kg)", None),
            "cout_moule": row.get("Coût moule (€)", None),
            "masse": row.get("Masse (kg)", None),
            "interp_points": json.dumps([[int(q), float(p)] for q, p in params.get("interp_points", []) if isinstance(q, (int, float)) and isinstance(p, (int, float))])
            })

    # Ecrasement de toutes les anciennes données du worksheet
    worksheet.clear()
    worksheet.update([sauvegarde[0].keys()] + [list(d.values()) for d in sauvegarde])


def get_comp_key(row):
        return f"{row.get('Ensemble','')}/{row.get('Sous-Ensemble','')}/{row.get('Composant','')}/{row.get('Fournisseur','')}".strip().lower()



# Titre principal de l'application
st.title("Estimation du coût de revient d’un véhicule en fonction de la quantité")
st.markdown("Version: v1")

# 1. Chargement de la nomenclature depuis Google Sheets

st.markdown("## 1. Chargement de la nomenclature")

@st.cache_data(ttl=10)
def charger_nomenclature_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
    worksheet = sheet.worksheet(st.secrets["sheets"]["worksheet_nomenclature"])
    data = worksheet.get_all_records()
    df = pd.DataFrame(data)

    # Nettoyage et normalisation
    df = df.dropna(subset=["Composant"], how='all')
    if "Masse unitaire" in df.columns and "Masse (kg)" in df.columns:
        df = df.drop(columns=["Masse (kg)"])
    if "Masse unitaire" in df.columns:
        df.rename(columns={"Masse unitaire": "Masse (kg)"}, inplace=True)
    if "Loi spécifique" not in df.columns:
        df["Loi spécifique"] = "Global"
    if "Masse (kg)" not in df.columns:
        df["Masse (kg)"] = None
    if "Prix matière" in df.columns and "Prix matière (€/kg)" not in df.columns:
        df.rename(columns={"Prix matière": "Prix matière (€/kg)"}, inplace=True)
    if "Prix matière (€/kg)" not in df.columns:
        df["Prix matière (€/kg)"] = None
    if "Coût moule" in df.columns and "Coût moule (€)" not in df.columns:
        df.rename(columns={"Coût moule": "Coût moule (€)"}, inplace=True)
    if "Coût moule (€)" not in df.columns:
        df["Coût moule (€)"] = None

    # Conversion propre des colonnes numériques avec virgules en points
    colonnes_numeriques = [
        "Masse (kg)", "Prix matière (€/kg)", 
        "Coût moule (€)", "Prix Effectif / Véhicule"
    ]
    for col in colonnes_numeriques:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).replace("€", "").replace(",", ".").replace(" ", "").strip())
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=10)
def charger_reglages_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(st.secrets["sheets"]["spreadsheet_id"])
    worksheet = sheet.worksheet(st.secrets["sheets"]["worksheet_reglages"])
    
    records = worksheet.get_all_records()

    # Reconstruction des paramètres
    comp_params = {}
    for ligne in records:
        comp_key = ligne["comp_key"].strip().lower()
        try:
            interp_points = json.loads(ligne["interp_points"])
        except Exception:
            interp_points = []
        comp_params[comp_key] = {
            "law": ligne.get("law", "Global"),
            "prix_matiere": ligne.get("prix_matiere"),
            "cout_moule": ligne.get("cout_moule"),
            "masse": ligne.get("masse"),
            "interp_points": interp_points
        }

    return comp_params


# Chargement effectif
df = charger_nomenclature_gsheet()
st.session_state.df_nomenclature = df.copy()

# Chargement des réglages sauvegardés dans Google Sheets
try:
    comp_params = charger_reglages_gsheet()
    st.session_state.comp_params = comp_params
    st.session_state.json_loaded = True

    # Mise à jour de la nomenclature avec les réglages
    for comp_key, params in comp_params.items():
        mask = df.apply(lambda row: f"{row.get('Ensemble','')}/{row.get('Sous-Ensemble','')}/{row.get('Composant','')}/{row.get('Fournisseur','')}".strip().lower() == comp_key, axis=1)
        df.loc[mask, "Prix matière (€/kg)"] = params.get("prix_matiere", None)
        df.loc[mask, "Coût moule (€)"] = params.get("cout_moule", None)
        df.loc[mask, "Masse (kg)"] = params.get("masse", None)
        df.loc[mask, "Loi spécifique"] = params.get("law", "Global")

    st.success("Paramètres rechargés depuis Google Sheets !")

except Exception as e:
    st.warning(f"Erreur lors du chargement des paramètres Google Sheets : {e}")
        

# 2. Consultation et modification de la nomenclature
st.markdown("## 2. Consultation et modification de la nomenclature")
st.write("Vous pouvez éditer le tableau ci-dessous : ajouter/modifier/supprimer des composants si besoin.")
st.write("- **Loi spécifique** : vous pouvez définir une loi d’interpolation personnalisée (quantité → prix unitaire) pour certains composants si vous disposez de devis ou d’historiques.")
st.write("- **Masse (kg), Prix matière (€/kg), Coût moule (€)** : pour les composants **moulés** (fournis par *Formes & Volumes* ou *Stratiforme Industries*), renseignez ces valeurs pour un calcul de coût unitaire basé sur la matière et l'amortissement du moule.")
# Note explicative pour les composants moulés
st.info("Pour les composants moulés, le coût unitaire sera calculé comme : **Prix matière × Masse unitaire + Coût moule ÷ Quantité totale produite**. Veillez à renseigner ces champs pour ces composants.")

# Affichage du tableau éditable dans un formulaire pour valider les modifications en une fois
with st.form(key="edit_form"):
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Loi spécifique": st.column_config.SelectboxColumn(
                "Loi spécifique",
                options=["Global", "Interpolation"]
            ),
            "Prix matière (€/kg)": st.column_config.NumberColumn(
                "Prix matière (€/kg)",
                help="Prix de la matière première en € par kg"
            ),
            "Coût moule (€)": st.column_config.NumberColumn(
                "Coût moule (€)",
                help="Coût du moule (€) pour ce composant (investissement outillage)"
            )
        }
    )
    submit = st.form_submit_button("Valider les modifications")
if submit:
    st.session_state.df_nomenclature = edited_df

    # Synchronisation de comp_params avec normalisation des clés
    st.session_state.comp_params = {}
    for _, row in edited_df.iterrows():
        if pd.isna(row.get("Composant")) or str(row.get("Composant")).strip() == "":
            continue  # Ignorer lignes vides

        comp_key = get_comp_key(row)
        st.session_state.comp_params[comp_key] = {
            "law": str(row.get("Loi spécifique", "Global")),
            "prix_matiere": row.get("Prix matière (€/kg)", None),
            "cout_moule": row.get("Coût moule (€)", None),
            "masse": row.get("Masse (kg)", None)
        }

    try:
        sauvegarder_parametres_gsheet()
        st.success("Modifications sauvegardées dans Google Sheets !")
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde : {e}")
else:
    edited_df = st.session_state.df_nomenclature


# 3. Choix du scénario de production
st.markdown("## 3. Choix du scénario de production")
st.write("Sélectionnez un nombre de véhicules à produire. Utilisez un raccourci ou entrez une valeur personnalisée.")

col1, col2 = st.columns([2, 3])
with col1:
    preset = st.radio("Raccourcis :", options=[10, 100, 500, 1000, "Autre"], index=1, horizontal=True)
with col2:
    if preset == "Autre":
        N = st.number_input("Nombre de véhicules (personnalisé)", min_value=1, step=1, value=52)
    else:
        N = preset

global_law = "Interpolation"
# Initialisation des paramètres globaux par défaut (stockés en session pour persistance)
if "comp_params" not in st.session_state:
    st.session_state.comp_params = {}
# Champs de paramétrage en fonction de la loi globale sélectionnée + rappel de la formule
if global_law == "Interpolation":
    # Points d'interpolation pour la loi globale
    if "interp_points" not in st.session_state:
        st.session_state.interp_points = pd.DataFrame({
            "Quantité": [1, 10, 100, 1000],
            "Facteur coût unitaire": [1.0, 0.85, 0.65, 0.5]

        })
    st.write("Le coût unitaire sera déterminé par interpolation linéaire à partir de points de référence définis (quantité vs facteur de coût unitaire relatif).")
    if st.button("Définir les points d'interpolation", key="btn_define_points"):
        @st.dialog("Points d'interpolation – Coût unitaire relatif")
        def interp_dialog():
            st.write("**Définissez des points (quantité produite vs facteur de coût unitaire par rapport au coût de base)** :")
            # Table des points d'interpolation éditable
            interp_df = st.data_editor(
                st.session_state.interp_points,
                num_rows="dynamic", use_container_width=True, hide_index=True,
                column_config={
                    "Quantité": st.column_config.NumberColumn("Quantité", min_value=1, step=1),
                    "Facteur coût unitaire": st.column_config.NumberColumn("Facteur coût unitaire", min_value=0.0, max_value=1.0, step=0.01)
                }
            )
            # Conseils d'utilisation
            st.markdown("*(Exemple : 1 → 1.0 signifie un coût de base à 1 unité; 1000 → 0.5 signifie un coût unitaire réduit à 50% du prix de base à 1000 unités.)*")
            if st.button("Enregistrer", key="save_interp_points"):
                # Trier par quantité et sauvegarder
                interp_df = interp_df.sort_values("Quantité")
                st.session_state.interp_points = interp_df
                try:
                    sauvegarder_parametres_gsheet()
                    st.success("Points d'interpolation globaux sauvegardés dans Google Sheets !")
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde : {e}")
                st.rerun()
        interp_dialog()
    # Affichage des points actuels en résumé
    if not st.session_state.interp_points.empty:
        st.write("Points d'interpolation actuels :")
        st.table(st.session_state.interp_points)
    st.write("*(Le coût unitaire pour une quantité N sera interpolé linéairement entre les points fournis, et restera constant en dehors de la plage définie.)*")

# 4. Calcul des coûts (production = N véhicules) si le tableau n'est pas vide
if not edited_df.empty:
    df_calc = st.session_state.df_nomenclature.copy()
    # Conversion des colonnes Quantité et Prix en numériques (NaN -> 0)
    df_calc["Quantité / Véhicule"] = pd.to_numeric(df_calc["Quantité / Véhicule"], errors='coerce').fillna(0)
    df_calc["Prix Effectif / Véhicule"] = pd.to_numeric(df_calc["Prix Effectif / Véhicule"], errors='coerce').fillna(0)
    df_calc["Masse (kg)"] = pd.to_numeric(df_calc["Masse (kg)"], errors='coerce')
    df_calc["Prix matière (€/kg)"] = pd.to_numeric(df_calc["Prix matière (€/kg)"], errors='coerce')
    df_calc["Coût moule (€)"] = pd.to_numeric(df_calc["Coût moule (€)"], errors='coerce')
    
    # Fonction utilitaire pour générer une clé unique identifiant un composant (pour stockage des paramètres spécifiques)
    def get_comp_key(row):
        return f"{row.get('Ensemble','')}/{row.get('Sous-Ensemble','')}/{row.get('Composant','')}/{row.get('Fournisseur','')}".strip().lower()
    
    results = []
    missing_data_parts = []  # liste des composants moulés avec données incomplètes
    total_per_vehicle = 0.0
    total_cost_moules = 0.0 

    for _, row in df_calc.iterrows():
        # Ignorer les lignes sans composant (ex : lignes ajoutées vides)
        if pd.isna(row["Composant"]) or str(row["Composant"]).strip() == "":
            continue
        Q = float(row["Quantité / Véhicule"]) if not pd.isna(row["Quantité / Véhicule"]) else 0
        if Q <= 0:
            continue  # ignorer si quantité n'est pas positive
        base_cost_vehicle = float(row["Prix Effectif / Véhicule"]) if not pd.isna(row["Prix Effectif / Véhicule"]) else 0.0
        base_cost_part = base_cost_vehicle / Q if Q != 0 else 0.0  # coût de base par pièce

        # Déterminer la loi à appliquer pour ce composant
        law = str(row.get("Loi spécifique", "Global"))
        if law is None or law == "" or law.lower() == "global":
            law = global_law  # utiliser la loi globale par défaut
        new_price_part = None

        # Vérifier si composant moulé 
        prix_matiere = row.get("Prix matière (€/kg)")
        cout_moule = row.get("Coût moule (€)")
        masse = row.get("Masse (kg)")
        is_moule = (
            not pd.isna(prix_matiere)
            and not pd.isna(cout_moule)
            and not pd.isna(masse)
        )
        if is_moule:
            try:
                masse = float(row["Masse (kg)"])
                prix_matiere = float(row["Prix matière (€/kg)"])
                cout_moule = float(row["Coût moule (€)"])
                new_price_part = masse * prix_matiere
                total_cost_moules += cout_moule
            except Exception:
                new_price_part = None
                missing_data_parts.append(str(row.get("Composant")))


        # Si aucun coût n'a encore été calculé -> appliquer la loi de décroissance classique
        if new_price_part is None:
            comp_key = get_comp_key(row)
            if str(row.get("Loi spécifique", "")).strip().lower() == "interpolation":
                # Loi spécifique avec points (à ajouter dans l’étape suivante)
                if comp_key not in st.session_state.comp_params:
                    st.session_state.comp_params[comp_key] = {
                        "law": "Interpolation",
                        "interp_points": [(1, base_cost_part), (1000, base_cost_part * 0.5)]
                    }
                elif "interp_points" not in st.session_state.comp_params[comp_key]:
                    st.session_state.comp_params[comp_key]["interp_points"] = [(1, base_cost_part), (1000, base_cost_part * 0.5)]

                interp_points = st.session_state.comp_params[comp_key]["interp_points"]



                if interp_points:
                    pts = sorted(interp_points, key=lambda x: x[0])
                    if N <= pts[0][0]:
                        new_price_part = pts[0][1]
                    elif N >= pts[-1][0]:
                        new_price_part = pts[-1][1]
                    else:
                        for i in range(len(pts) - 1):
                            q1, p1 = pts[i]
                            q2, p2 = pts[i+1]
                            if q1 <= N <= q2:
                                ratio = (N - q1) / (q2 - q1)
                                new_price_part = p1 + ratio * (p2 - p1)
                                break
                else:
                    new_price_part = base_cost_part
            else:
                # Loi globale interpolation
                points = st.session_state.interp_points.sort_values("Quantité")
                if points.empty:
                    new_price_part = base_cost_part
                else:
                    pts = list(zip(points["Quantité"], points["Facteur coût unitaire"]))
                    if N <= pts[0][0]:
                        factor = float(pts[0][1])
                    elif N >= pts[-1][0]:
                        factor = float(pts[-1][1])
                    else:
                        for i in range(len(pts) - 1):
                            q_low, fac_low = float(pts[i][0]), float(pts[i][1])
                            q_high, fac_high = float(pts[i+1][0]), float(pts[i+1][1])
                            if N <= q_high:
                                factor = fac_low + (fac_high - fac_low) * ((N - q_low) / (q_high - q_low))
                                break
                    new_price_part = base_cost_part * factor


        # Calcul du coût par véhicule pour ce composant
        new_cost_vehicle = new_price_part * Q
        total_per_vehicle += new_cost_vehicle
        results.append({
            "Ensemble": row.get("Ensemble"),
            "Sous-Ensemble": row.get("Sous-Ensemble"),
            "Composant": row.get("Composant"),
            "Quantité / Véhicule": int(Q) if Q.is_integer() else Q,
            "Fournisseur": row.get("Fournisseur"),
            "Prix unitaire estimé (€)": new_price_part,
            "Coût par véhicule (€)": new_cost_vehicle,
            "Is_moule": is_moule 
        })

    # 5. Ajustement des paramètres des composants (optionnel)
    st.markdown("## 4. Ajustement des paramètres des composants (optionnel)")
    st.write("Ajustez les paramètres des lois de décroissance pour des composants spécifiques si nécessaire :")
    if "popup_open_for" not in st.session_state:
        st.session_state.popup_open_for = None
        st.session_state.popup_comp_name = None

    specific_law_rows = []
    for idx, row in df_calc.iterrows():
        law_spec = str(row.get("Loi spécifique", ""))
        if law_spec and law_spec.lower() not in ["", "global", "fixe"] and not pd.isna(row.get("Composant")) and str(row.get("Composant")).strip() != "":
            specific_law_rows.append((idx, row["Composant"], law_spec))
    if specific_law_rows:
        for idx, comp_name, law in specific_law_rows:
            comp_key = get_comp_key(df_calc.loc[idx])
            if st.button(f"Ajuster paramètres : {comp_name} ({law})", key=f"btn_param_{idx}"):
                st.session_state.popup_open_for = comp_key
                st.session_state.popup_comp_name = comp_name
                st.rerun()
    else:
        st.write("*(Aucun composant avec loi spécifique nécessitant un ajustement de paramètres.)*")
    
    if st.session_state.popup_open_for is not None:
        comp_key = st.session_state.popup_open_for
        comp_name = st.session_state.popup_comp_name

        # Toujours détecter la loi actuelle à partir du tableau
        mask = st.session_state.df_nomenclature.apply(
            lambda row: f"{row.get('Ensemble','')}/{row.get('Sous-Ensemble','')}/{row.get('Composant','')}/{row.get('Fournisseur','')}".strip().lower() == comp_key,
            axis=1
        )
        detected_law = "Global"
        if mask.any():
            row_df = st.session_state.df_nomenclature[mask].iloc[0]
            detected_law = str(row_df.get("Loi spécifique", "Global")).strip()

        # Initialiser comp_params si nécessaire
        if comp_key not in st.session_state.comp_params:
            if detected_law.lower() == "interpolation":
                st.session_state.comp_params[comp_key] = {
                    "law": "Interpolation",
                    "interp_points": [(1, 1.0), (1000, 0.5)]
                }
            else:
                st.session_state.comp_params[comp_key] = {"law": detected_law}

        # Enforcer la bonne loi dans comp_params (synchro sécurité)
        st.session_state.comp_params[comp_key]["law"] = detected_law

        law = st.session_state.comp_params[comp_key].get("law", "Interpolation")

        def show_param_dialog():
            @st.dialog(f"Paramètres pour {comp_name} – Loi {law}")
            def param_dialog():
                if law == "Interpolation":

                    interp = st.session_state.comp_params[comp_key].get("interp_points", [])
                    if not interp:
                        mask = st.session_state.df_nomenclature.apply(
                            lambda row: f"{row.get('Ensemble','')}/{row.get('Sous-Ensemble','')}/{row.get('Composant','')}/{row.get('Fournisseur','')}".strip().lower() == comp_key,
                            axis=1
                        )
                        prix_base = 1.0
                        if mask.any():
                            row_df = st.session_state.df_nomenclature[mask].iloc[0]
                            try:
                                prix_effectif = float(row_df.get("Prix Effectif / Véhicule", 1.0))
                                quantite = float(row_df.get("Quantité / Véhicule", 1))
                                prix_base = prix_effectif / quantite if quantite > 0 else prix_effectif
                            except Exception:
                                prix_base = 1.0
                        interp = [(1, round(prix_base, 2)), (1000, round(prix_base / 2, 2))]
                        st.session_state.comp_params[comp_key]["interp_points"] = interp


                    df_interp = pd.DataFrame(interp, columns=["Quantité", "Prix unitaire (€)"])
                    df_interp_edited = st.data_editor(
                        df_interp,
                        num_rows="dynamic",
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Quantité": st.column_config.NumberColumn("Quantité", min_value=1, step=1),
                            "Prix unitaire (€)": st.column_config.NumberColumn("Prix unitaire (€)", min_value=0.0, step=0.01),
                        }
                    )
                    if st.button("Valider", key="val_interp_points_global_popup"):
                        st.session_state.comp_params[comp_key]["interp_points"] = df_interp_edited.dropna().sort_values("Quantité").values.tolist()
                        
                        # On sauvegarde les paramètres
                        try:
                            sauvegarder_parametres_gsheet()
                            st.success("Paramètres sauvegardés dans Google Sheets !")
                        except Exception as e:
                            st.error(f"Erreur lors de la sauvegarde : {e}")

                        # On ferme la popup et on relance
                        st.session_state.popup_open_for = None
                        st.session_state.popup_comp_name = None
                        st.rerun()
            param_dialog()

        show_param_dialog()


    # 6. Résultats (production = N véhicules)
    st.markdown(f"## 5. Résultats (production = {N} véhicules)")
    if not results:
        st.warning("Aucun composant à calculer. Vérifiez le tableau de nomenclature.")
    else:
        df_res = pd.DataFrame(results)
        # Arrondir les valeurs monétaires à 2 décimales
        df_res["Prix unitaire estimé (€)"] = df_res["Prix unitaire estimé (€)"].astype(float).round(2)
        df_res["Coût par véhicule (€)"] = df_res["Coût par véhicule (€)"].astype(float).round(2)

        # Identifier les comp_keys avec loi spécifique
        specific_keys = set()
        for idx, row in df_calc.iterrows():
            law_spec = str(row.get("Loi spécifique", ""))
            if law_spec and law_spec.lower() not in ["", "global", "fixe"]:
                specific_keys.add(get_comp_key(row))

        # Liste des indices des composants moulés (à partir de df_res)
        mouled_indices = df_res[df_res["Is_moule"]].index

        # Liste des indices des composants avec loi spécifique
        specific_indices = [
            idx for idx, row in df_res.iterrows()
            if get_comp_key(row) in specific_keys
        ]

        # Supprimer la colonne Is_moule pour ne pas l'afficher
        df_display = df_res.drop(columns=["Is_moule"])

        # Coloration des lignes
        def highlight_row(row):
            idx = row.name
            if idx in mouled_indices:
                return ['background-color: #ccffcc'] * len(row)
            elif idx in specific_indices:
                return ['background-color: #ffffcc'] * len(row)
            else:
                return [''] * len(row)

        df_display_styled = df_display.style.apply(highlight_row, axis=1)
        st.write(df_display_styled)

        # Affichage des coûts totaux
        total_per_vehicle = round(total_per_vehicle, 2)
        total_all = round(total_per_vehicle * N, 2)
        st.markdown(f"**Coût total par véhicule :** {total_per_vehicle:,.2f} €")
        st.markdown(f"**Coût total pour {N} véhicules :** {total_all:,.2f} €")
        if total_cost_moules > 0:
            total_all_with_moules = total_all + total_cost_moules
            st.markdown(f"**Investissement moules (fixe) :** {total_cost_moules:,.2f} €")
            st.markdown(f"**Coût total pour {N} véhicules, moules compris :** {total_all_with_moules:,.2f} €")


        # Avertissement si données de moulage manquantes
        if missing_data_parts:
            st.warning(
                "Composants moulés sans données matière/moule complètes (loi globale appliquée à la place) : "
                + ", ".join(set(missing_data_parts))
            )

        # Légende des couleurs
        st.markdown(
            "<p style='font-size:0.9em'>"
            "<span style='background-color:#ccffcc;'>&nbsp;&nbsp;&nbsp;</span> <strong>Vert</strong> : calcul matière+moule appliqué (composant moulé) &nbsp;&nbsp; "
            "<span style='background-color:#ffffcc;'>&nbsp;&nbsp;&nbsp;</span> <strong>Jaune</strong> : loi spécifique appliquée</p>",
            unsafe_allow_html=True
        )
