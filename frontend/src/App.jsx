import { useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Simple SVG Icons
const Icons = {
  bolt: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  ),
  text: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
      <polyline points="4 7 4 4 20 4 20 7" />
      <line x1="9" y1="20" x2="15" y2="20" />
      <line x1="12" y1="4" x2="12" y2="20" />
    </svg>
  ),
  brain: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="4" />
    </svg>
  ),
  book: (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
      <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
    </svg>
  ),
  target: (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="6" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  ),
  cpu: (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" />
      <line x1="9" y1="1" x2="9" y2="4" />
      <line x1="15" y1="1" x2="15" y2="4" />
      <line x1="9" y1="20" x2="9" y2="23" />
      <line x1="15" y1="20" x2="15" y2="23" />
    </svg>
  ),
  keyboard: (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <rect x="2" y="4" width="20" height="16" rx="2" />
      <line x1="6" y1="8" x2="6" y2="8" />
      <line x1="10" y1="8" x2="10" y2="8" />
      <line x1="14" y1="8" x2="14" y2="8" />
      <line x1="18" y1="8" x2="18" y2="8" />
      <line x1="6" y1="16" x2="18" y2="16" />
    </svg>
  ),
  github: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
    </svg>
  ),
  arrow: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  ),
}

// Transliteration map
const TRANSLIT_MAP = {
  'aa': 'आ', 'a': 'अ', 'ii': 'ई', 'ee': 'ई', 'i': 'इ',
  'uu': 'ऊ', 'oo': 'ऊ', 'u': 'उ', 'ri': 'ऋ', 'e': 'ए',
  'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
  'kaa': 'का', 'ka': 'क', 'k': 'क्',
  'khaa': 'खा', 'kha': 'ख', 'kh': 'ख्',
  'gaa': 'गा', 'ga': 'ग', 'g': 'ग्',
  'ghaa': 'घा', 'gha': 'घ', 'gh': 'घ्',
  'chaa': 'चा', 'cha': 'च', 'ch': 'च्',
  'jaa': 'जा', 'ja': 'ज', 'j': 'ज्',
  'jhaa': 'झा', 'jha': 'झ', 'jh': 'झ्',
  'Taa': 'टा', 'Ta': 'ट', 'T': 'ट्',
  'Thaa': 'ठा', 'Tha': 'ठ', 'Th': 'ठ्',
  'Daa': 'डा', 'Da': 'ड', 'D': 'ड्',
  'Dhaa': 'ढा', 'Dha': 'ढ', 'Dh': 'ढ्',
  'taa': 'ता', 'ta': 'त', 't': 'त्',
  'thaa': 'था', 'tha': 'थ', 'th': 'थ्',
  'daa': 'दा', 'da': 'द', 'd': 'द्',
  'dhaa': 'धा', 'dha': 'ध', 'dh': 'ध्',
  'naa': 'ना', 'na': 'न', 'n': 'न्',
  'paa': 'पा', 'pa': 'प', 'p': 'प्',
  'phaa': 'फा', 'pha': 'फ', 'ph': 'फ्',
  'baa': 'बा', 'ba': 'ब', 'b': 'ब्',
  'bhaa': 'भा', 'bha': 'भ', 'bh': 'भ्',
  'maa': 'मा', 'ma': 'म', 'm': 'म्',
  'yaa': 'या', 'ya': 'य', 'y': 'य्',
  'raa': 'रा', 'ra': 'र', 'r': 'र्',
  'laa': 'ला', 'la': 'ल', 'l': 'ल्',
  'vaa': 'वा', 'va': 'व', 'v': 'व्',
  'waa': 'वा', 'wa': 'व', 'w': 'व्',
  'shaa': 'शा', 'sha': 'श', 'sh': 'श्',
  'saa': 'सा', 'sa': 'स', 's': 'स्',
  'haa': 'हा', 'ha': 'ह', 'h': 'ह्',
  'ksha': 'क्ष', 'gya': 'ज्ञ', 'tra': 'त्र', 'shri': 'श्री',
}

function transliterate(text) {
  if (/[\u0900-\u097F]/.test(text)) return text
  let result = '', i = 0
  const lower = text.toLowerCase()
  while (i < lower.length) {
    let matched = false
    for (let len = 4; len >= 1; len--) {
      const chunk = lower.slice(i, i + len)
      if (TRANSLIT_MAP[chunk]) {
        result += TRANSLIT_MAP[chunk]
        i += len
        matched = true
        break
      }
    }
    if (!matched) { result += text[i]; i++ }
  }
  return result.replace(/्([ा-ौ])/g, '$1').replace(/्$/g, '')
}

function isDevanagari(text) {
  return /[\u0900-\u097F]/.test(text)
}

const EXAMPLES = ['bharat', 'namaste', 'vidyalaya', 'aanandam', 'shiksha']

function App() {
  const [word, setWord] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!word.trim()) return
    setLoading(true)
    setError(null)
    const devanagariWord = transliterate(word.trim())
    try {
      const res = await fetch(`${API_URL}/segment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word: devanagariWord })
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to segment')
      }
      const data = await res.json()
      setResult({
        ...data,
        originalInput: word.trim(),
        wasTransliterated: !isDevanagari(word.trim())
      })
    } catch (err) {
      setError(err.message)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const preview = word.trim() && !isDevanagari(word.trim()) ? transliterate(word.trim()) : null

  return (
    <div className="page">
      {/* Header */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-brand">
            <span className="nav-logo">अ</span>
            <span className="nav-title">Syllabify</span>
          </div>
          <div className="nav-links">
            <a href="#hero">Home</a>
            <a href="#tool">Tool</a>
            <a href="#about">About</a>
            <a href="https://github.com/akanupam/hyphenification-and-syllabification-of-indian-scripts" target="_blank" rel="noopener noreferrer" className="nav-github">
              {Icons.github}
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero" id="hero">
        <div className="hero-container">
          <h1>देवनागरी Syllabifier</h1>
          <p className="hero-desc">
            An ML-powered tool to break down Devanagari words into syllables.
            Built using a BiLSTM+CRF model trained on curated Hindi text data.
          </p>
          <div className="hero-features">
            <div className="feature">
              <span className="feature-icon">{Icons.bolt}</span>
              <span>Instant Results</span>
            </div>
            <div className="feature">
              <span className="feature-icon">{Icons.text}</span>
              <span>Transliteration</span>
            </div>
            <div className="feature">
              <span className="feature-icon">{Icons.brain}</span>
              <span>ML Powered</span>
            </div>
          </div>
          <a href="#tool" className="hero-cta">Try It Now</a>
        </div>
      </section>

      {/* Tool Section */}
      <section className="tool-section" id="tool">
        <div className="tool-container">
          <div className="tool-header">
            <h2>Try It Out</h2>
            <p>Type in English (romanized) or directly in देवनागरी</p>
          </div>

          <form className="input-section" onSubmit={handleSubmit}>
            <div className="input-wrapper">
              <input
                type="text"
                value={word}
                onChange={(e) => setWord(e.target.value)}
                placeholder="Type a word..."
              />
              {preview && <div className="preview">{Icons.arrow} {preview}</div>}
            </div>
            <button type="submit" disabled={loading || !word.trim()}>
              {loading ? '...' : 'Split'}
            </button>
          </form>

          <div className="examples-row">
            <span className="examples-label">Try:</span>
            {EXAMPLES.map((ex, i) => (
              <button key={i} className="example-btn" type="button" onClick={() => setWord(ex)}>
                {ex}
              </button>
            ))}
          </div>

          {error && <div className="error">{error}</div>}

          {result ? (
            <div className="result">
              {result.wasTransliterated && (
                <div className="result-original">{result.originalInput} {Icons.arrow}</div>
              )}
              <div className="result-word">{result.word}</div>

              <div className="syllables">
                {result.syllables.map((s, i) => (
                  <span key={i} className="syllable">{s}</span>
                ))}
              </div>

              <div className="result-hyphenated">{result.hyphenated}</div>

              <div className="result-details">
                <div className="detail">
                  <span className="label">Syllables</span>
                  <span className="value">{result.count}</span>
                </div>
                <div className="detail">
                  <span className="label">Characters</span>
                  <span className="value">{result.word.length}</span>
                </div>
                <div className="detail">
                  <span className="label">Input Type</span>
                  <span className="value">{result.wasTransliterated ? 'Roman' : 'Hindi'}</span>
                </div>
              </div>
            </div>
          ) : !error && (
            <div className="empty-state">Enter a word above to see syllable breakdown</div>
          )}
        </div>
      </section>

      {/* About Section */}
      <section className="about" id="about">
        <div className="about-container">
          <h2>What is Syllabification?</h2>
          <div className="about-grid">
            <div className="about-card">
              <span className="about-icon">{Icons.book}</span>
              <h3>Definition</h3>
              <p>
                Syllabification divides words into syllables — the basic units of pronunciation.
                In Devanagari, syllables typically consist of consonant clusters followed by vowels.
              </p>
            </div>
            <div className="about-card">
              <span className="about-icon">{Icons.target}</span>
              <h3>Use Cases</h3>
              <p>
                Text-to-speech systems, spell checkers, language learning tools,
                typography, text justification, poetry analysis, and linguistic research.
              </p>
            </div>
            <div className="about-card">
              <span className="about-icon">{Icons.cpu}</span>
              <h3>How It Works</h3>
              <p>
                Uses a BiLSTM+CRF (Bidirectional LSTM with Conditional Random Field)
                model trained on curated Hindi data to predict syllable boundaries.
              </p>
            </div>
            <div className="about-card">
              <span className="about-icon">{Icons.keyboard}</span>
              <h3>Transliteration</h3>
              <p>
                Type using English letters (like "bharat" for भारत) and the tool
                automatically converts it to Devanagari before processing.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-container">
          <div className="footer-brand">
            <span className="nav-logo">अ</span>
            <span>Devanagari Syllabifier</span>
          </div>
          <div className="footer-tech">

          </div>
          <div className="footer-links">
            <a href="#tool">Tool</a>
            <a href="#about">About</a>
            <a href="https://github.com/akanupam/hyphenification-and-syllabification-of-indian-scripts" target="_blank" rel="noopener noreferrer">{Icons.github}</a>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
