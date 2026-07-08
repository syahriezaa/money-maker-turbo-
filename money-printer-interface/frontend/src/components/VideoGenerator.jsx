import React, { useState } from 'react';
import { Sparkles, Play, Video, AlignLeft, Volume2, Globe, Layers, AlertCircle, CheckCircle2, Sliders } from 'lucide-react';
import { generateScript } from '../api';

const VOICES = [
  { name: "en-US-GuyNeural", label: "Guy (English US - Male)", lang: "en" },
  { name: "en-US-JennyNeural", label: "Jenny (English US - Female)", lang: "en" },
  { name: "en-GB-SoniaNeural", label: "Sonia (English UK - Female)", lang: "en" },
  { name: "en-GB-RyanNeural", label: "Ryan (English UK - Male)", lang: "en" },
  { name: "en-US-Standard", label: "English Narrator (Auto-Cast Local Chatterbox)", lang: "en" },
  { name: "en-chatterbox-male", label: "English Deep Male (Local Chatterbox)", lang: "en" },
  { name: "en-chatterbox-female", label: "English Expressive Female (Local Chatterbox)", lang: "en" },
  { name: "id-ID-Standard", label: "Indonesian Narrator (Auto-Cast Local Chatterbox)", lang: "id" },
  { name: "id-chatterbox-male", label: "Indonesian Deep Male (Local Chatterbox)", lang: "id" },
  { name: "id-chatterbox-female", label: "Indonesian Expressive Female (Local Chatterbox)", lang: "id" },
  { name: "es-ES-AlvaroNeural", label: "Alvaro (Spanish - Male)", lang: "es" },
  { name: "fr-FR-HenriNeural", label: "Henri (French - Male)", lang: "fr" }
];

const LANGUAGES = [
  { code: "en", label: "English" },
  { code: "id", label: "Indonesian" },
  { code: "es", label: "Spanish" },
  { code: "fr", label: "French" },
  { code: "de", label: "German" },
  { code: "zh", label: "Chinese" }
];

const MOCK_SCRIPTS = {
  space: "Did you know that space is completely silent? That's because sound waves need a medium like air to travel, and space is a vacuum. Also, one day on Venus is longer than a whole year on Earth. Venus spins so slowly that it takes 243 Earth days to make one rotation, but only 225 Earth days to orbit the Sun! Truly mind-blowing space mechanics.",
  ocean: "The deep ocean is one of the most mysterious places on our planet. More people have stood on the moon than have explored the Mariana Trench, the deepest spot in the ocean. In these dark depths, creatures have evolved bizarre adaptations like bioluminescence, glowing in the pitch black to lure prey or find mates.",
  history: "In 1932, Australia declared war on emus. The military deployed soldiers armed with machine guns to combat crop-destroying emus in Western Australia, but the emus proved surprisingly tactical and evasive, winning the conflict. This remains one of the most bizarre military failures in history.",
  general: "Automated content generation is transforming how creators reach audiences. By combining generative AI scripting with voice synthesis, high-engaging video formats can be rendered in seconds. This allows publishers to focus on ideation and strategy while machines do the heavy rendering lifework."
};

export default function VideoGenerator({ onSubmitTask }) {
  const [subject, setSubject] = useState("");
  const [aspectRatio, setAspectRatio] = useState("9:16");
  const [voiceName, setVoiceName] = useState("en-US-GuyNeural");
  const [language, setLanguage] = useState("en");
  const [paragraphs, setParagraphs] = useState(2);
  
  // Script Editor states
  const [scriptText, setScriptText] = useState("");
  const [isGeneratingScript, setIsGeneratingScript] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState(null);

  // Local AI Generation Overrides
  const [showLocalSettings, setShowLocalSettings] = useState(false);
  const [localSteps, setLocalSteps] = useState("");
  const [localCfg, setLocalCfg] = useState("");
  const [localSeed, setLocalSeed] = useState("");
  const [localNegativePrompt, setLocalNegativePrompt] = useState("");

  // Generate Script using AI from backend
  const handleGenerateScript = async () => {
    if (!subject) {
      setMessage({ type: 'error', text: "Please enter a subject first to write a script." });
      return;
    }

    setIsGeneratingScript(true);
    setMessage(null);

    try {
      const data = await generateScript({
        subject: subject,
        language: language,
        paragraphs: paragraphs
      });

      const scriptSource = data.script || "";

      // Stream script into state
      let currentWordIndex = 0;
      const words = scriptSource.split(" ");
      setScriptText("");
      
      const interval = setInterval(() => {
        if (currentWordIndex < words.length) {
          setScriptText(prev => prev + (prev ? " " : "") + words[currentWordIndex]);
          currentWordIndex++;
        } else {
          clearInterval(interval);
          setIsGeneratingScript(false);
          setMessage({ type: 'success', text: "Script drafted by AI! Feel free to edit below." });
        }
      }, 50);
    } catch (err) {
      setIsGeneratingScript(false);
      setMessage({ type: 'error', text: `Failed to generate script: ${err.message}` });
    }
  };

  // Preview TTS voice
  const handlePreviewVoice = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const textToSpeak = scriptText || "This is a preview of the voice synthesis narration.";
      const utterance = new SpeechSynthesisUtterance(textToSpeak.slice(0, 140)); // read first part
      utterance.rate = 1.0;
      
      // Attempt to load native browser voices that match lang
      const browserVoices = window.speechSynthesis.getVoices();
      const match = browserVoices.find(v => v.lang.startsWith(language));
      if (match) {
        utterance.voice = match;
      }
      
      window.speechSynthesis.speak(utterance);
    } else {
      setMessage({ type: 'error', text: "Web speech preview not supported in this browser." });
    }
  };

  // Submit Task to Backend
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!subject) {
      setMessage({ type: 'error', text: "Please fill out the Video Subject/Topic." });
      return;
    }

    if (!scriptText.trim()) {
      setMessage({ type: 'error', text: "Please generate or write your script in the editor first before rendering." });
      return;
    }

    setIsSubmitting(true);
    setMessage(null);

    // If script is custom edited, we'll embed the script text inside the subject payload
    const finalSubject = scriptText 
      ? `Topic: ${subject}. Content Script: ${scriptText}`
      : subject;

    try {
      await onSubmitTask({
        video_subject: finalSubject,
        video_aspect_ratio: aspectRatio,
        voice_name: voiceName,
        language: language,
        paragraph_number: paragraphs,
        local_steps: localSteps ? parseInt(localSteps) : null,
        local_cfg: localCfg ? parseFloat(localCfg) : null,
        local_seed: localSeed ? parseInt(localSeed) : null,
        local_negative_prompt: localNegativePrompt || null
      });
      
      setMessage({ 
        type: 'success', 
        text: "Task successfully initialized! Transitioning to render tracker." 
      });
      
      // Clear form
      setSubject("");
      setScriptText("");
    } catch (err) {
      setMessage({ type: 'error', text: `Failed to queue task: ${err.message}` });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="tab-content fade-in">
      <div className="dashboard-main-grid">
        {/* Left Column: Input Form */}
        <div className="glass-card">
          <h2 className="section-title">
            <Video size={20} /> Create Video Task
          </h2>

          {message && (
            <div className={`alert-banner ${message.type === 'error' ? 'error' : ''}`}>
              {message.type === 'error' ? <AlertCircle size={18} /> : <CheckCircle2 size={18} />}
              <span>{message.text}</span>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label>Video Subject / Topic Prompt</label>
              <input 
                type="text" 
                placeholder="e.g. 5 Shocking Facts about Space, Deep Ocean Creatures, Bizarre History..."
                value={subject}
                onChange={e => setSubject(e.target.value)}
                disabled={isSubmitting}
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Language</label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <select 
                    value={language} 
                    onChange={e => setLanguage(e.target.value)}
                    disabled={isSubmitting}
                  >
                    {LANGUAGES.map(lang => (
                      <option key={lang.code} value={lang.code}>{lang.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label>Narration Voice</label>
                <select 
                  value={voiceName} 
                  onChange={e => setVoiceName(e.target.value)}
                  disabled={isSubmitting}
                >
                  {VOICES.filter(v => v.lang === language).map(voice => (
                    <option key={voice.name} value={voice.name}>{voice.label}</option>
                  ))}
                  {VOICES.filter(v => v.lang !== language).map(voice => (
                    <option key={voice.name} value={voice.name}>{voice.label} (Non-native)</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Paragraph Count (Length)</label>
                <select 
                  value={paragraphs} 
                  onChange={e => setParagraphs(parseInt(e.target.value))}
                  disabled={isSubmitting}
                >
                  {[1, 2, 3, 4, 5, 10, 15, 20, 25, 30].map(num => {
                    const seconds = num * 20;
                    const minutes = Math.floor(seconds / 60);
                    const remainingSeconds = seconds % 60;
                    const timeStr = minutes > 0 
                      ? `${minutes}m${remainingSeconds > 0 ? ` ${remainingSeconds}s` : ''}`
                      : `${seconds}s`;
                    return (
                      <option key={num} value={num}>{num} Paragraph(s) (~{timeStr})</option>
                    );
                  })}
                </select>
              </div>

              <div className="form-group">
                <label>Preview Narrator Voice</label>
                <button 
                  type="button" 
                  className="btn btn-secondary" 
                  onClick={handlePreviewVoice}
                  style={{ height: '46px' }}
                >
                  <Volume2 size={16} /> Listen Audio Preview
                </button>
              </div>
            </div>

            <div className="form-group">
              <label>Visual Aspect Ratio</label>
              <div className="aspect-ratio-group">
                <div 
                  className={`aspect-option ${aspectRatio === '9:16' ? 'selected' : ''}`}
                  onClick={() => setAspectRatio("9:16")}
                >
                  <div className="aspect-box portrait" />
                  <span className="aspect-label">9:16 Portrait</span>
                  <span className="aspect-desc">TikTok, YouTube Shorts, Reels</span>
                </div>

                <div 
                  className={`aspect-option ${aspectRatio === '16:9' ? 'selected' : ''}`}
                  onClick={() => setAspectRatio("16:9")}
                >
                  <div className="aspect-box wide" />
                  <span className="aspect-label">16:9 Landscape</span>
                  <span className="aspect-desc">YouTube standard, Widescreen</span>
                </div>
              </div>
            </div>

            {/* Collapsible Local Image settings overrides */}
            <div style={{ marginTop: '16px', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '10px', overflow: 'hidden' }}>
              <button
                type="button"
                onClick={() => setShowLocalSettings(!showLocalSettings)}
                style={{
                  width: '100%',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  background: 'rgba(0,0,0,0.2)',
                  border: 'none',
                  padding: '12px 16px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '0.9rem',
                  fontWeight: 600
                }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sliders size={16} /> Advanced Local AI (ComfyUI) Settings
                </span>
                <span>{showLocalSettings ? '▲' : '▼'}</span>
              </button>

              {showLocalSettings && (
                <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '14px', background: 'rgba(0,0,0,0.1)' }}>
                  <div className="form-row">
                    <div className="form-group">
                      <label>Steps Override</label>
                      <input
                        type="number"
                        placeholder="Leave blank for default"
                        value={localSteps}
                        onChange={e => setLocalSteps(e.target.value)}
                        style={{ color: 'white' }}
                      />
                    </div>

                    <div className="form-group">
                      <label>CFG Override</label>
                      <input
                        type="number"
                        step="0.5"
                        placeholder="Leave blank for default"
                        value={localCfg}
                        onChange={e => setLocalCfg(e.target.value)}
                        style={{ color: 'white' }}
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label>Seed Override</label>
                    <input
                      type="number"
                      placeholder="Leave blank for random"
                      value={localSeed}
                      onChange={e => setLocalSeed(e.target.value)}
                      style={{ color: 'white' }}
                    />
                  </div>

                  <div className="form-group">
                    <label>Negative Prompt Override</label>
                    <textarea
                      placeholder="Leave blank for default negative prompt"
                      value={localNegativePrompt}
                      onChange={e => setLocalNegativePrompt(e.target.value)}
                      style={{ minHeight: '50px', padding: '8px', fontSize: '0.8rem', color: 'black', borderRadius: '6px' }}
                    />
                  </div>
                </div>
              )}
            </div>

            {!scriptText.trim() ? (
              <button 
                type="button" 
                className="btn btn-secondary" 
                style={{ width: '100%', marginTop: '12px', background: 'var(--color-primary)', color: 'white' }}
                onClick={handleGenerateScript}
                disabled={isGeneratingScript || isSubmitting}
              >
                {isGeneratingScript ? (
                  <>
                    <Sparkles size={18} className="spin-loader" /> Step 1: Drafting Script...
                  </>
                ) : (
                  <>
                    <Sparkles size={18} /> Step 1: Generate Script for Review
                  </>
                )}
              </button>
            ) : (
              <button 
                type="submit" 
                className="btn btn-primary" 
                style={{ width: '100%', marginTop: '12px' }}
                disabled={isSubmitting || isGeneratingScript}
              >
                {isSubmitting ? (
                  <>
                    <Sparkles size={18} className="spin-loader" /> Step 2: Rendering Video...
                  </>
                ) : (
                  <>
                    <Play size={18} /> Step 2: Render Video from Script
                  </>
                )}
              </button>
            )}
          </form>
        </div>

        {/* Right Column: AI Script Editor */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 className="section-title">
            <AlignLeft size={20} /> AI Script Editor
          </h2>
          
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.5', marginBottom: '16px' }}>
            Review, edit, and fine-tune your narration script here. The script must be drafted or written before you can render the video.
          </p>

          <div className="script-editor-container" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div className="script-editor-header">
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>Drafting Tool</span>
              <button 
                type="button" 
                className="editor-btn primary"
                onClick={handleGenerateScript}
                disabled={isGeneratingScript || isSubmitting}
              >
                {isGeneratingScript ? (
                  <>
                    <Sparkles size={12} className="spin-loader" /> Writing...
                  </>
                ) : (
                  <>
                    <Sparkles size={12} /> Auto-Write Script
                  </>
                )}
              </button>
            </div>

            <textarea
              className="script-textarea"
              placeholder="Start typing your script here, or click 'Auto-Write Script' to draft it using generative AI..."
              value={scriptText}
              onChange={e => setScriptText(e.target.value)}
              disabled={isGeneratingScript || isSubmitting}
              style={{ flex: 1 }}
            />

            <div className="script-editor-footer">
              <span>Words: {scriptText ? scriptText.split(/\s+/).filter(Boolean).length : 0}</span>
              <span>Characters: {scriptText.length}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
