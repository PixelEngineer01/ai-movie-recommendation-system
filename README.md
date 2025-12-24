

# 🎬 AI Movie Recommendation System

A **Netflix-style movie recommendation system** built using **Machine Learning** and deployed with **Streamlit**.
The application recommends movies based on **content similarity**, supports **smart search**, **genre-based browsing**, and displays **similarity scores** in a clean and intuitive UI.

---

## ✨ Features

* 🔍 **Smart Movie Search**
  Handles typos, partial movie names, and case differences using fuzzy matching.

* 🎭 **Genre-Based Recommendations**
  Browse movies by genre or explore randomly using *All Genres*.

* 🧠 **Content-Based Recommendation Engine**
  Uses **TF-IDF vectorization** and **Cosine Similarity** for accurate recommendations.

* 📊 **Similarity Score (%)**
  Shows how closely each recommended movie matches the selected movie.

* 🎬 **Movie Posters**
  Posters are fetched in real-time using the **OMDb API** and cached for speed.

* ⚡ **Fast & Optimized**
  No large similarity matrix stored. Similarity is computed on demand.

* 🎨 **Netflix-like UI**
  Clean, responsive interface with compact movie cards.

* 🔄 **Reset Button**
  Instantly clears search and genre selections.

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **TF-IDF Vectorizer**
* **Cosine Similarity**
* **RapidFuzz**
* **OMDb API**
* **TMDB Dataset**

---

## ⚙️ How It Works 

1. Movie metadata is converted into numerical vectors using **TF-IDF**.
2. When a user searches for a movie:

   * The system finds the closest matching title using **smart fuzzy matching**.
   * Cosine similarity is calculated between the selected movie and others.
3. The most similar movies are ranked and shown with similarity scores.
4. Movie posters are fetched from the OMDb API and displayed in the UI.

---

## 🏗️ Architecture Diagram

User → Streamlit UI → Fuzzy Search
     → TF-IDF + Cosine Similarity
     → Recommendations + Scores
     → OMDb API → Posters

---
## 🚀 Running the App Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/PixelEngineer01/ai-movie-recommendation-system.git
   cd ai-movie-recommendation-system
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your OMDb API key:

   ```bash
   .streamlit/secrets.toml
   ```

   ```toml
   OMDB_API_KEY = "your_api_key_here"
   ```

4. Run the app:

   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment

The application is deployed using **Streamlit Cloud** with secure secrets management.
No API keys or large model files are exposed in the repository.

---

## 📌 Project Highlights

* No `similarity.pkl` file (memory efficient)
* Dynamic similarity thresholding
* Smart fuzzy search for better accuracy
* Clean GitHub workflow
* Interview & resume ready project

---

## 🎯 Future Improvements

* Autocomplete movie search
* Hybrid recommendation (content + popularity)
* User-based personalization
* Mobile-first UI enhancements

---

## 👨‍💻 Author

**Ankush Mahato**
Interested in Machine Learning, AI, and building real-world applications.
***Email:work.ankushmahato@gmail.com***