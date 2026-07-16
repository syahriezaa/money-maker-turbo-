import React, { useState, useEffect } from 'react';
import { Eye, EyeOff, Save, ShieldAlert, CheckCircle, Sliders, Volume2, Key, Type } from 'lucide-react';
import { getConfig, updateConfig } from '../api';

export default function Settings() {
  const [config, setConfig] = useState({
    llm_provider: "openai",
    openai_api_key: "",
    gemini_api_key: "",
    groq_api_key: "",
    deepseek_api_key: "",
    tts_provider: "edge-tts",
    voice_name: "en-US-GuyNeural",
    pexels_api_key: "",
    subtitles_enabled: true,
    subtitle_color: "#FFFFFF",
    subtitle_fontsize: 24,
    output_dir: "./output",
    local_steps: 20,
    local_cfg: 7.5,
    local_seed: 1337,
    local_negative_prompt: "low quality, worst quality, deformed, bad anatomy, bad hands, blurry, watermark, text, signature"
  });

  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState(null);
  
  // Show/Hide password keys state
  const [showKeys, setShowKeys] = useState({
    openai: false,
    gemini: false,
    groq: false,
    deepseek: false,
    pexels: false
  });

  useEffect(() => {
    async function loadConfigData() {
      try {
        const data = await getConfig();
        setConfig(prev => ({ ...prev, ...data }));
      } catch (err) {
        setMessage({ type: 'error', text: "Failed to fetch settings config." });
      } finally {
        setIsLoading(false);
      }
    }
    loadConfigData();
  }, []);

  const handleChange = (key, value) => {
    setConfig(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const toggleKeyVisibility = (key) => {
    setShowKeys(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSaving(true);
    setMessage(null);

    try {
      await updateConfig(config);
      setMessage({ type: 'success', text: "Configuration saved successfully!" });
      setTimeout(() => setMessage(null), 3000);
    } catch (err) {
      setMessage({ type: 'error', text: `Failed to save configuration: ${err.message}` });
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className="tab-content fade-in">
        <div className="glass-card" style={{ textAlign: 'center', padding: '40px' }}>
          <Sliders size={32} className="spin-loader" style={{ color: 'var(--color-primary)' }} />
          <p style={{ marginTop: '12px', color: 'var(--text-secondary)' }}>Loading settings config...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="tab-content fade-in">
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        
        {message && (
          <div className={`alert-banner ${message.type === 'error' ? 'error' : ''}`}>
            {message.type === 'error' ? <ShieldAlert size={18} /> : <CheckCircle size={18} />}
            <span>{message.text}</span>
          </div>
        )}

        <div className="dashboard-main-grid" style={{ gridTemplateColumns: '1.2fr 1fr' }}>
          {/* Left Column: API Keys and Model Settings */}
          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <h2 className="section-title">
              <Key size={18} /> AI Language Model (LLM) & API Keys
            </h2>

            <div className="form-group">
              <label>Default LLM Provider</label>
              <select 
                value={config.llm_provider}
                onChange={e => handleChange('llm_provider', e.target.value)}
              >
                <option value="openai">OpenAI (GPT-4o, GPT-3.5)</option>
                <option value="gemini">Google Gemini Pro</option>
                <option value="groq">Groq Llama-3 (Ultra-Fast)</option>
                <option value="deepseek">DeepSeek Coder & Chat (V3/R1)</option>
              </select>
            </div>

            <div className="form-group">
              <label>OpenAI API Key</label>
              <div className="api-key-input-group">
                <input 
                  type={showKeys.openai ? "text" : "password"} 
                  value={config.openai_api_key}
                  onChange={e => handleChange('openai_api_key', e.target.value)}
                  placeholder="sk-proj-..."
                />
                <button 
                  type="button" 
                  className="api-key-toggle-visibility"
                  onClick={() => toggleKeyVisibility('openai')}
                >
                  {showKeys.openai ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <div className="form-group">
              <label>Google Gemini API Key</label>
              <div className="api-key-input-group">
                <input 
                  type={showKeys.gemini ? "text" : "password"} 
                  value={config.gemini_api_key}
                  onChange={e => handleChange('gemini_api_key', e.target.value)}
                  placeholder="AIzaSy..."
                />
                <button 
                  type="button" 
                  className="api-key-toggle-visibility"
                  onClick={() => toggleKeyVisibility('gemini')}
                >
                  {showKeys.gemini ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <div className="form-group">
              <label>Groq Cloud API Key</label>
              <div className="api-key-input-group">
                <input 
                  type={showKeys.groq ? "text" : "password"} 
                  value={config.groq_api_key}
                  onChange={e => handleChange('groq_api_key', e.target.value)}
                  placeholder="gsk_..."
                />
                <button 
                  type="button" 
                  className="api-key-toggle-visibility"
                  onClick={() => toggleKeyVisibility('groq')}
                >
                  {showKeys.groq ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <div className="form-group">
              <label>DeepSeek API Key</label>
              <div className="api-key-input-group">
                <input 
                  type={showKeys.deepseek ? "text" : "password"} 
                  value={config.deepseek_api_key}
                  onChange={e => handleChange('deepseek_api_key', e.target.value)}
                  placeholder="sk-..."
                />
                <button 
                  type="button" 
                  className="api-key-toggle-visibility"
                  onClick={() => toggleKeyVisibility('deepseek')}
                >
                  {showKeys.deepseek ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <h2 className="section-title" style={{ marginTop: '16px' }}>
              <Volume2 size={18} /> Voice Narrator Defaults
            </h2>

            <div className="form-row">
              <div className="form-group">
                <label>TTS Provider</label>
                <select 
                  value={config.tts_provider}
                  onChange={e => handleChange('tts_provider', e.target.value)}
                >
                  <option value="edge-tts">Microsoft Edge TTS (Free)</option>
                  <option value="elevenlabs">ElevenLabs Premium (High Fidelity)</option>
                  <option value="local-chatterbox">Local Chatterbox Turbo</option>
                  <option value="local-fishaudio">Local Fish Speech (s2-pro)</option>
                </select>
              </div>

              <div className="form-group">
                <label>Narration Voice</label>
                <input 
                  type="text" 
                  value={config.voice_name}
                  onChange={e => handleChange('voice_name', e.target.value)}
                  placeholder="e.g. en-US-GuyNeural"
                />
              </div>
            </div>
          </div>

          {/* Right Column: Sourcing, Subtitles and Output settings */}
          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <h2 className="section-title">
              <Type size={18} /> Subtitles & Sourcing Engines
            </h2>

            <div className="form-group">
              <label>Pexels Stock Video API Key</label>
              <div className="api-key-input-group">
                <input 
                  type={showKeys.pexels ? "text" : "password"} 
                  value={config.pexels_api_key}
                  onChange={e => handleChange('pexels_api_key', e.target.value)}
                  placeholder="Enter Pexels API Key..."
                />
                <button 
                  type="button" 
                  className="api-key-toggle-visibility"
                  onClick={() => toggleKeyVisibility('pexels')}
                >
                  {showKeys.pexels ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                Required to download high quality stock footage. Get a free key on pexels.com
              </span>
            </div>

            <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '12px', background: 'rgba(0,0,0,0.15)', padding: '12px', borderRadius: '10px', marginTop: '8px' }}>
              <input 
                type="checkbox" 
                id="subtitles_enabled"
                checked={config.subtitles_enabled}
                onChange={e => handleChange('subtitles_enabled', e.target.checked)}
                style={{ width: '18px', height: '18px', accentColor: 'var(--color-primary)', cursor: 'pointer' }}
              />
              <label htmlFor="subtitles_enabled" style={{ cursor: 'pointer', select: 'none' }}>
                Enable Overlay Subtitles on Video File
              </label>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Subtitle Font Color</label>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <input 
                    type="color" 
                    value={config.subtitle_color}
                    onChange={e => handleChange('subtitle_color', e.target.value)}
                    style={{ border: 'none', background: 'none', width: '40px', height: '40px', cursor: 'pointer' }}
                  />
                  <input 
                    type="text" 
                    value={config.subtitle_color}
                    onChange={e => handleChange('subtitle_color', e.target.value)}
                    style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}
                  />
                </div>
              </div>

              <div className="form-group">
                <label>Subtitle Font Size</label>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                  <input 
                    type="range" 
                    min="14" 
                    max="48" 
                    value={config.subtitle_fontsize}
                    onChange={e => handleChange('subtitle_fontsize', parseInt(e.target.value))}
                    style={{ flex: 1, accentColor: 'var(--color-primary)' }}
                  />
                  <span style={{ fontSize: '0.9rem', fontWeight: 'bold', width: '32px', textAlign: 'right' }}>
                    {config.subtitle_fontsize}px
                  </span>
                </div>
              </div>
            </div>

            <h2 className="section-title" style={{ marginTop: '16px' }}>
              <Sliders size={18} /> Storage & Target Settings
            </h2>

            <div className="form-group">
              <label>Local Render Directory</label>
              <input 
                type="text" 
                value={config.output_dir}
                onChange={e => handleChange('output_dir', e.target.value)}
                placeholder="e.g. ./output"
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                Target directory on host system where rendered files will build.
              </span>
            </div>

            <h2 className="section-title" style={{ marginTop: '16px' }}>
              <Sliders size={18} /> Local AI Image Generator (ComfyUI Schema)
            </h2>

            <div className="form-row">
              <div className="form-group">
                <label>Sampling Steps</label>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                  <input 
                    type="range" 
                    min="1" 
                    max="50" 
                    value={config.local_steps || 20}
                    onChange={e => handleChange('local_steps', parseInt(e.target.value))}
                    style={{ flex: 1, accentColor: 'var(--color-primary)' }}
                  />
                  <span style={{ fontSize: '0.9rem', fontWeight: 'bold', width: '32px', textAlign: 'right' }}>
                    {config.local_steps || 20}
                  </span>
                </div>
              </div>

              <div className="form-group">
                <label>CFG Scale</label>
                <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                  <input 
                    type="range" 
                    min="1" 
                    max="20" 
                    step="0.5"
                    value={config.local_cfg || 7.5}
                    onChange={e => handleChange('local_cfg', parseFloat(e.target.value))}
                    style={{ flex: 1, accentColor: 'var(--color-primary)' }}
                  />
                  <span style={{ fontSize: '0.9rem', fontWeight: 'bold', width: '32px', textAlign: 'right' }}>
                    {config.local_cfg || 7.5}
                  </span>
                </div>
              </div>
            </div>

            <div className="form-group">
              <label>Default Seed</label>
              <input 
                type="number" 
                value={config.local_seed || 1337}
                onChange={e => handleChange('local_seed', parseInt(e.target.value) || 1337)}
                placeholder="e.g. 1337"
              />
            </div>

            <div className="form-group">
              <label>Default Negative Prompt</label>
              <textarea 
                value={config.local_negative_prompt || ""}
                onChange={e => handleChange('local_negative_prompt', e.target.value)}
                placeholder="low quality, worst quality..."
                style={{ minHeight: '60px', padding: '8px', fontSize: '0.85rem', color: 'black', borderRadius: '8px' }}
              />
            </div>
          </div>
        </div>

        <div className="glass-card" style={{ display: 'flex', justifyContent: 'flex-end', padding: '16px 24px' }}>
          <button 
            type="submit" 
            className="btn btn-primary" 
            disabled={isSaving}
            style={{ width: 'fit-content' }}
          >
            {isSaving ? (
              <>
                <Save size={18} className="spin-loader" /> Saving Config...
              </>
            ) : (
              <>
                <Save size={18} /> Save All Settings
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
