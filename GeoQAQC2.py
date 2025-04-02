import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import base64
import re

# Configuration de la page
st.set_page_config(
    page_title="GeoQAQC",
    page_icon="📊",
    layout="wide"
)

# Fonction pour créer un identifiant valide à partir d'un nom de colonne
def make_valid_id(column_name):
    # Remplacer les caractères non alphanumériques par des underscores
    return re.sub(r'\W+', '_', column_name).lower()

# Fonction pour mapper les colonnes
def map_columns(df, mapping_dict):
    # Créer un nouveau DataFrame avec les colonnes mappées
    mapped_df = pd.DataFrame()
    
    for target_col, source_col in mapping_dict.items():
        if source_col in df.columns:
            mapped_df[target_col] = df[source_col]
    
    return mapped_df

# Auteur et informations
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/examples/app/geospatial_app/thumbnail.png", width=100)
    st.markdown("### GeoQAQC")
    st.markdown("*Contrôle Qualité des Analyses Chimiques des Roches*")
    st.markdown("---")
    st.markdown("**Auteur:** Didier Ouedraogo, P.Geo")
    st.markdown("**Version:** 1.0.0")
    
    # Section d'aide
    with st.expander("Guide d'utilisation"):
        st.markdown("""
        **Comment utiliser cette application:**
        
        1. **Type de Contrôle**: Sélectionnez d'abord le type d'analyse que vous souhaitez effectuer.
        2. **Importation des Données**: Téléchargez ou collez vos données.
        3. **Mappage des Colonnes**: Associez les colonnes de vos données aux champs requis.
        4. **Analyse**: Générez et visualisez les résultats.
        
        Pour toute question, contactez l'auteur.
        """)

# Titre principal
st.title("GeoQAQC")
st.markdown("### Contrôle Qualité des Analyses Chimiques des Roches")

# Initialisation des états de session
if 'data' not in st.session_state:
    st.session_state.data = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'mapped_data' not in st.session_state:
    st.session_state.mapped_data = None
if 'mapping_done' not in st.session_state:
    st.session_state.mapping_done = False

# Onglets
tabs = st.tabs(["Type de Contrôle", "Importation des Données", "Mappage des Colonnes", "Analyse"])

with tabs[0]:
    st.header("Choisir le Type de Carte de Contrôle")
    
    control_type = st.selectbox(
        "Type de contrôle:",
        ["Standards CRM", "Blancs", "Duplicatas (nuage de points et régression)"],
        key="control_type"
    )
    
    if control_type == "Standards CRM":
        col1, col2 = st.columns(2)
        
        with col1:
            reference_value = st.number_input(
                "Valeur de référence:",
                min_value=0.0,
                step=0.0001,
                format="%.4f",
                key="reference_value"
            )
            
            reference_stddev = st.number_input(
                "Écart-type de référence:",
                min_value=0.0,
                step=0.0001,
                format="%.4f",
                key="reference_stddev"
            )
        
        with col2:
            tolerance_type = st.radio(
                "Type de tolérance:",
                ["Pourcentage (%)", "Multiple de l'écart-type"],
                key="tolerance_type"
            )
            
            if tolerance_type == "Pourcentage (%)":
                tolerance_value = st.number_input(
                    "Tolérance (%):",
                    min_value=0.0,
                    max_value=100.0,
                    value=10.0,
                    step=0.1,
                    key="tolerance_percent"
                )
            else:
                tolerance_value = st.number_input(
                    "Multiple de l'écart-type:",
                    min_value=0.0,
                    value=2.0,
                    step=0.1,
                    key="tolerance_stddev"
                )
        
        # Définir les champs requis pour l'analyse
        st.session_state.required_fields = {
            "sample_id": "Identifiant de l'échantillon",
            "measured_value": "Valeur mesurée"
        }
    
    elif control_type == "Blancs":
        # Définir les champs requis pour l'analyse des blancs
        st.session_state.required_fields = {
            "sample_id": "Identifiant du blanc",
            "measured_value": "Valeur mesurée"
        }
    
    elif control_type == "Duplicatas (nuage de points et régression)":
        # Définir les champs requis pour l'analyse des duplicatas
        st.session_state.required_fields = {
            "original_value": "Valeur originale",
            "duplicate_value": "Valeur dupliquée"
        }
    
    # Bouton pour passer à l'onglet suivant
    if st.button("Continuer vers l'importation des données", key="continue_to_import"):
        st.session_state.active_tab = 1
        st.experimental_rerun()

# Fonction pour calculer les limites pour les CRM
def calculate_crm_limits(reference_value, tolerance_type, tolerance_value, reference_stddev=None):
    if tolerance_type == "Pourcentage (%)":
        tolerance = tolerance_value / 100
        upper_limit = reference_value * (1 + tolerance)
        lower_limit = reference_value * (1 - tolerance)
    else:  # Multiple de l'écart-type
        if reference_stddev is None or reference_stddev == 0:
            st.error("L'écart-type de référence doit être défini et supérieur à zéro pour utiliser ce type de tolérance.")
            return None, None
        upper_limit = reference_value + (tolerance_value * reference_stddev)
        lower_limit = reference_value - (tolerance_value * reference_stddev)
    
    return lower_limit, upper_limit

# Dans le deuxième onglet - Importation des données
with tabs[1]:
    st.header("Importer les Données")
    
    import_method = st.radio(
        "Méthode d'importation:",
        ["Téléchargement de fichier", "Copier-coller des données"],
        key="import_method"
    )
    
    st.session_state.mapping_done = False  # Réinitialiser l'état de mappage
    
    if import_method == "Téléchargement de fichier":
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv", "txt"])
        
        if uploaded_file is not None:
            separator = st.selectbox(
                "Séparateur:",
                [",", ";", "Tab"],
                key="file_separator"
            )
            
            sep_dict = {",": ",", ";": ";", "Tab": "\t"}
            try:
                if separator == "Tab":
                    df = pd.read_csv(uploaded_file, sep="\t")
                else:
                    df = pd.read_csv(uploaded_file, sep=separator)
                
                st.session_state.data = df
                st.success(f"Fichier chargé avec succès! {len(df)} lignes et {len(df.columns)} colonnes.")
                st.write("Aperçu des données:")
                st.dataframe(df.head())
                
                # Bouton pour continuer
                if st.button("Continuer vers le mappage des colonnes", key="continue_to_mapping_file"):
                    st.session_state.active_tab = 2
                    st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")
    else:
        pasted_data = st.text_area(
            "Collez vos données (format CSV ou tableau séparé par des tabulations):",
            height=200,
            key="pasted_data"
        )
        
        separator = st.selectbox(
            "Séparateur:",
            [",", ";", "Tab"],
            key="paste_separator"
        )
        
        if st.button("Traiter les données"):
            if pasted_data:
                sep_dict = {",": ",", ";": ";", "Tab": "\t"}
                try:
                    df = pd.read_csv(StringIO(pasted_data), sep=sep_dict[separator])
                    st.session_state.data = df
                    st.success(f"Données traitées avec succès! {len(df)} lignes et {len(df.columns)} colonnes.")
                    st.write("Aperçu des données:")
                    st.dataframe(df.head())
                    
                    # Bouton pour continuer
                    if st.button("Continuer vers le mappage des colonnes", key="continue_to_mapping_paste"):
                        st.session_state.active_tab = 2
                        st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement des données: {e}")
            else:
                st.warning("Veuillez coller des données avant de les traiter.")

# Dans le troisième onglet - Mappage des colonnes
with tabs[2]:
    st.header("Mappage des Colonnes")
    
    if st.session_state.data is None:
        st.warning("Aucune donnée n'a été importée. Veuillez d'abord importer des données dans l'onglet 'Importation des Données'.")
    else:
        df = st.session_state.data
        
        st.write("Associez les colonnes de vos données aux champs requis par l'application:")
        
        # Création du formulaire de mappage
        mapping_form = st.form("column_mapping_form")
        
        with mapping_form:
            mapping_dict = {}
            
            for field_id, field_name in st.session_state.required_fields.items():
                mapping_dict[field_id] = st.selectbox(
                    f"Champ '{field_name}':",
                    options=["-- Sélectionner une colonne --"] + list(df.columns),
                    key=f"mapping_{field_id}"
                )
            
            submit_button = st.form_submit_button("Appliquer le mappage")
        
        if submit_button:
            # Vérifier que toutes les colonnes ont été mappées
            if all(col != "-- Sélectionner une colonne --" for col in mapping_dict.values()):
                # Créer un dictionnaire de mappage inversé
                inverse_mapping = {v: k for k, v in mapping_dict.items()}
                
                # Créer un DataFrame mappé
                mapped_data = pd.DataFrame()
                
                for source_col, target_field in inverse_mapping.items():
                    mapped_data[target_field] = df[source_col]
                
                st.session_state.mapped_data = mapped_data
                st.session_state.column_mapping = mapping_dict
                st.session_state.mapping_done = True
                
                st.success("Mappage des colonnes effectué avec succès!")
                st.write("Aperçu des données mappées:")
                st.dataframe(mapped_data.head())
                
                # Bouton pour continuer
                if st.button("Continuer vers l'analyse", key="continue_to_analysis"):
                    st.session_state.active_tab = 3
                    st.experimental_rerun()
            else:
                st.error("Veuillez associer toutes les colonnes requises.")

# Dans le quatrième onglet - Analyse
with tabs[3]:
    st.header("Analyse des Données")
    
    if not st.session_state.mapping_done:
        st.warning("Veuillez d'abord mapper les colonnes de vos données dans l'onglet 'Mappage des Colonnes'.")
    else:
        data = st.session_state.mapped_data
        control_type = st.session_state.control_type
        
        # Analyse selon le type de contrôle
        if control_type == "Standards CRM":
            if st.button("Générer la Carte de Contrôle"):
                # Préparation des données
                id_column = "sample_id"
                value_column = "measured_value"
                
                analysis_data = data[[id_column, value_column]].copy()
                analysis_data = analysis_data.dropna()
                analysis_data[value_column] = pd.to_numeric(analysis_data[value_column], errors='coerce')
                analysis_data = analysis_data.dropna()
                
                if analysis_data.empty:
                    st.error("Aucune donnée numérique valide trouvée pour l'analyse.")
                else:
                    # Récupération des paramètres
                    reference_value = st.session_state.reference_value
                    tolerance_type = st.session_state.tolerance_type
                    
                    if tolerance_type == "Pourcentage (%)":
                        tolerance_value = st.session_state.tolerance_percent
                    else:
                        tolerance_value = st.session_state.tolerance_stddev
                        
                    reference_stddev = st.session_state.reference_stddev if 'reference_stddev' in st.session_state else 0
                    
                    # Calcul des limites
                    lower_limit, upper_limit = calculate_crm_limits(
                        reference_value,
                        tolerance_type,
                        tolerance_value,
                        reference_stddev
                    )
                    
                    if lower_limit is not None and upper_limit is not None:
                        # Statistiques
                        values = analysis_data[value_column].values
                        mean = np.mean(values)
                        std_dev = np.std(values)
                        min_val = np.min(values)
                        max_val = np.max(values)
                        
                        # Création du graphique avec Plotly
                        fig = go.Figure()
                        
                        # Données mesurées
                        fig.add_trace(go.Scatter(
                            x=analysis_data[id_column],
                            y=analysis_data[value_column],
                            mode='lines+markers',
                            name='Valeur mesurée',
                            line=dict(color='rgb(75, 192, 192)', width=2),
                            marker=dict(size=8)
                        ))
                        
                        # Valeur de référence
                        fig.add_trace(go.Scatter(
                            x=analysis_data[id_column],
                            y=[reference_value] * len(analysis_data),
                            mode='lines',
                            name='Valeur référence',
                            line=dict(color='rgb(54, 162, 235)', width=2, dash='dash')
                        ))
                        
                        # Limites
                        fig.add_trace(go.Scatter(
                            x=analysis_data[id_column],
                            y=[upper_limit] * len(analysis_data),
                            mode='lines',
                            name='Limite supérieure',
                            line=dict(color='rgb(255, 99, 132)', width=2, dash='dash')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=analysis_data[id_column],
                            y=[lower_limit] * len(analysis_data),
                            mode='lines',
                            name='Limite inférieure',
                            line=dict(color='rgb(255, 99, 132)', width=2, dash='dash')
                        ))
                        
                        # Récupérer le nom original de la colonne pour le titre
                        original_id_column = st.session_state.column_mapping.get('sample_id', 'Identifiant')
                        original_value_column = st.session_state.column_mapping.get('measured_value', 'Valeur')
                        
                        # Mise en forme
                        fig.update_layout(
                            title=f"GeoQAQC - Carte de Contrôle CRM - {original_value_column}",
                            xaxis_title=original_id_column,
                            yaxis_title=original_value_column,
                            height=600,
                            hovermode="closest"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Tableau des statistiques
                        st.subheader("Statistiques")
                        
                        stats_col1, stats_col2 = st.columns(2)
                        
                        with stats_col1:
                            st.markdown(f"**Valeur de référence:** {reference_value:.4f}")
                            
                            if reference_stddev > 0:
                                st.markdown(f"**Écart-type de référence:** {reference_stddev:.4f}")
                            
                            if tolerance_type == "Pourcentage (%)":
                                st.markdown(f"**Tolérance:** {tolerance_value:.2f}%")
                            else:
                                st.markdown(f"**Tolérance:** {tolerance_value:.1f} × écart-type")
                        
                        with stats_col2:
                            st.markdown(f"**Moyenne:** {mean:.4f}")
                            st.markdown(f"**Écart-type:** {std_dev:.4f}")
                            st.markdown(f"**Min:** {min_val:.4f}")
                            st.markdown(f"**Max:** {max_val:.4f}")
                        
                        # Tableau de données
                        st.subheader("Résultats détaillés")
                        
                        # Création d'un DataFrame avec les résultats
                        results_df = analysis_data.copy()
                        results_df['Écart (%)'] = ((results_df[value_column] - reference_value) / reference_value) * 100
                        
                        if reference_stddev > 0:
                            results_df['Z-score'] = (results_df[value_column] - reference_value) / reference_stddev
                        
                        results_df['Statut'] = results_df[value_column].apply(
                            lambda x: 'OK' if lower_limit <= x <= upper_limit else 'Hors limites'
                        )
                        
                        # Renommer les colonnes du tableau de résultats avec les noms originaux
                        display_df = results_df.copy()
                        display_df.rename(columns={
                            'sample_id': original_id_column,
                            'measured_value': original_value_column
                        }, inplace=True)
                        
                        # Afficher le tableau avec coloration conditionnelle
                        st.dataframe(display_df.style.apply(
                            lambda x: ['background-color: #ffcccc' if v == 'Hors limites' else '' for v in x],
                            subset=['Statut']
                        ))
                        
                        # Bouton d'export
                        csv = display_df.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="geoqaqc_crm_results.csv">Télécharger les résultats (CSV)</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
        elif control_type == "Duplicatas (nuage de points et régression)":
            if st.button("Générer la Carte de Contrôle"):
                # Préparation des données
                original_column = "original_value"
                replicate_column = "duplicate_value"
                
                analysis_data = data[[original_column, replicate_column]].copy()
                analysis_data = analysis_data.dropna()
                analysis_data[original_column] = pd.to_numeric(analysis_data[original_column], errors='coerce')
                analysis_data[replicate_column] = pd.to_numeric(analysis_data[replicate_column], errors='coerce')
                analysis_data = analysis_data.dropna()
                
                if analysis_data.empty:
                    st.error("Aucune donnée numérique valide trouvée pour l'analyse.")
                else:
                    # Calcul de la régression linéaire
                    x = analysis_data[original_column].values
                    y = analysis_data[replicate_column].values
                    
                    slope, intercept = np.polyfit(x, y, 1)
                    r = np.corrcoef(x, y)[0, 1]
                    
                    # Calcul des statistiques
                    differences = np.abs(y - x)
                    mean_diff = np.mean(differences)
                    
                    relative_diff = np.abs(y - x) / ((x + y) / 2) * 100
                    mean_relative_diff = np.nanmean(relative_diff)
                    
                    # Récupérer les noms originaux des colonnes pour le titre
                    original_value_name = st.session_state.column_mapping.get('original_value', 'Valeur originale')
                    duplicate_value_name = st.session_state.column_mapping.get('duplicate_value', 'Valeur dupliquée')
                    
                    # Création du graphique avec Plotly
                    fig = go.Figure()
                    
                    # Nuage de points
                    fig.add_trace(go.Scatter(
                        x=analysis_data[original_column],
                        y=analysis_data[replicate_column],
                        mode='markers',
                        name='Duplicatas',
                        marker=dict(
                            color='rgb(75, 192, 192)',
                            size=10,
                            opacity=0.8
                        )
                    ))
                    
                    # Ligne de régression
                    x_range = np.linspace(min(x), max(x), 100)
                    y_pred = slope * x_range + intercept
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name=f'Régression linéaire (y = {slope:.4f}x + {intercept:.4f})',
                        line=dict(color='rgb(255, 99, 132)', width=2)
                    ))
                    
                    # Ligne d'égalité parfaite (y = x)
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=x_range,
                        mode='lines',
                        name='Ligne d\'égalité (y=x)',
                        line=dict(color='rgb(54, 162, 235)', width=2, dash='dash')
                    ))
                    
                    # Mise en forme
                    fig.update_layout(
                        title=f"GeoQAQC - Analyse des Duplicatas - {original_value_name} vs {duplicate_value_name}",
                        xaxis_title=original_value_name,
                        yaxis_title=duplicate_value_name,
                        height=600,
                        hovermode="closest"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau des statistiques
                    st.subheader("Statistiques")
                    
                    st.markdown(f"**Équation de régression:** y = {slope:.4f}x + {intercept:.4f}")
                    st.markdown(f"**Coefficient de corrélation (R²):** {r*r:.4f}")
                    st.markdown(f"**Différence absolue moyenne:** {mean_diff:.4f}")
                    st.markdown(f"**Différence relative moyenne:** {mean_relative_diff:.2f}%")
                    
                    # Tableau de données
                    st.subheader("Résultats détaillés")
                    
                    # Création d'un DataFrame avec les résultats
                    results_df = analysis_data.copy()
                    results_df['Diff. Abs.'] = np.abs(results_df[replicate_column] - results_df[original_column])
                    results_df['Diff. Rel. (%)'] = np.abs(results_df[replicate_column] - results_df[original_column]) / ((results_df[original_column] + results_df[replicate_column]) / 2) * 100
                    
                    # Renommer les colonnes pour affichage
                    display_df = results_df.copy()
                    display_df.rename(columns={
                        'original_value': original_value_name,
                        'duplicate_value': duplicate_value_name
                    }, inplace=True)
                    
                    st.dataframe(display_df)
                    
                    # Bouton d'export
                    csv = display_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="geoqaqc_duplicate_results.csv">Télécharger les résultats (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
        elif control_type == "Blancs":
            if st.button("Générer la Carte de Contrôle"):
                # Préparation des données
                id_column = "sample_id"
                value_column = "measured_value"
                
                analysis_data = data[[id_column, value_column]].copy()
                analysis_data = analysis_data.dropna()
                analysis_data[value_column] = pd.to_numeric(analysis_data[value_column], errors='coerce')
                analysis_data = analysis_data.dropna()
                
                if analysis_data.empty:
                    st.error("Aucune donnée numérique valide trouvée pour l'analyse.")
                else:
                    # Récupérer les noms originaux des colonnes pour le titre
                    original_id_column = st.session_state.column_mapping.get('sample_id', 'Identifiant')
                    original_value_column = st.session_state.column_mapping.get('measured_value', 'Valeur')
                
                    # Calcul des statistiques
                    values = analysis_data[value_column].values
                    mean = np.mean(values)
                    std_dev = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    # Limites de détection estimées
                    lod = mean + 3 * std_dev
                    
                    # Création du graphique avec Plotly
                    fig = go.Figure()
                    
                    # Données mesurées
                    fig.add_trace(go.Scatter(
                        x=analysis_data[id_column],
                        y=analysis_data[value_column],
                        mode='lines+markers',
                        name='Valeur mesurée',
                        line=dict(color='rgb(75, 192, 192)', width=2),
                        marker=dict(size=8)
                    ))
                    
                    # Moyenne
                    fig.add_trace(go.Scatter(
                        x=analysis_data[id_column],
                        y=[mean] * len(analysis_data),
                        mode='lines',
                        name='Moyenne',
                        line=dict(color='rgb(54, 162, 235)', width=2, dash='dash')
                    ))
                    
                    # Limite de détection
                    fig.add_trace(go.Scatter(
                        x=analysis_data[id_column],
                        y=[lod] * len(analysis_data),
                        mode='lines',
                        name='Limite de détection (LOD)',
                        line=dict(color='rgb(255, 99, 132)', width=2, dash='dash')
                    ))
                    
                    # Mise en forme
                    fig.update_layout(
                        title=f"GeoQAQC - Carte de Contrôle Blancs - {original_value_column}",
                        xaxis_title=original_id_column,
                        yaxis_title=original_value_column,
                        height=600,
                        hovermode="closest"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tableau des statistiques
                    st.subheader("Statistiques")
                    
                    st.markdown(f"**Moyenne:** {mean:.4f}")
                    st.markdown(f"**Écart-type:** {std_dev:.4f}")
                    st.markdown(f"**Min:** {min_val:.4f}")
                    st.markdown(f"**Max:** {max_val:.4f}")
                    st.markdown(f"**Limite de détection estimée (LOD):** {lod:.4f}")
                    
                    # Tableau de données
                    st.subheader("Résultats détaillés")
                    
                    # Création d'un DataFrame avec les résultats
                    results_df = analysis_data.copy()
                    results_df['Statut'] = results_df[value_column].apply(
                        lambda x: 'OK' if x <= lod else 'Élevé'
                    )
                    
                    # Renommer les colonnes pour affichage
                    display_df = results_df.copy()
                    display_df.rename(columns={
                        'sample_id': original_id_column,
                        'measured_value': original_value_column
                    }, inplace=True)
                    
                    # Afficher le tableau avec coloration conditionnelle
                    st.dataframe(display_df.style.apply(
                        lambda x: ['background-color: #ffcccc' if v == 'Élevé' else '' for v in x],
                        subset=['Statut']
                    ))
                    
                    # Bouton d'export
                    csv = display_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="geoqaqc_blank_results.csv">Télécharger les résultats (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)

# Gestion des onglets actifs
if 'active_tab' in st.session_state:
    # Trouver l'élément iframe qui contient l'onglet actif et exécuter un script JavaScript pour le cliquer
    active_tab = st.session_state.active_tab
    # Supprimer l'état pour éviter les boucles
    del st.session_state.active_tab
    
    st.markdown(f"""
    <script>
        document.addEventListener('DOMContentLoaded', (event) => {{
            // Attendez que la page soit chargée
            setTimeout(() => {{
                // Cliquez sur l'onglet approprié
                document.querySelectorAll('.stTabs button')[{active_tab}].click();
            }}, 100);
        }});
    </script>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**GeoQAQC** © 2025 - Développé par Didier Ouedraogo, P.Geo")