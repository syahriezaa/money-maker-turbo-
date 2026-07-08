import React, { useState, useEffect, useRef } from 'react';
import { Terminal, RefreshCw, XCircle, AlertCircle, PlayCircle, Loader2, Trash2 } from 'lucide-react';

// ────────────────────────────────────────────────────────────
// Komponen: SubProgressPanel
// Menampilkan data sub-progress real-time dari proses AI
// ────────────────────────────────────────────────────────────
function SubProgressPanel({ subProgress }) {
  // Jangan render apapun jika data tidak ada
  if (!subProgress) return null;

  const {
    phase,
    segment,
    total_segments,
    pct,
    step,
    total_steps,
    speed,
    eta_seconds,
  } = subProgress;

  // Peta ikon berdasarkan nama fase
  const phaseIcon = {
    'Sintesis Suara'      : '🎙️',
    'Generasi Musik AI'   : '🎵',
    'Generasi Gambar AI'  : '🖼️',
    'Render Video'        : '🎬',
  }[phase] || '⚙️';

  // Format ETA ke string yang mudah dibaca
  const formatEta = (seconds) => {
    if (seconds == null || seconds <= 0) return null;
    if (seconds >= 60) {
      const menit = Math.floor(seconds / 60);
      const detik = Math.round(seconds % 60);
      return detik > 0 ? `${menit} mnt ${detik} dtk` : `${menit} menit`;
    }
    return `~${Math.round(seconds)} dtk`;
  };

  const etaText  = formatEta(eta_seconds);
  const fillPct  = Math.min(Math.max(pct ?? 0, 0), 100);
  const hasSpeed = speed && speed > 0;
  const hasSegment = segment != null && total_segments != null;

  return (
    <div className="sub-progress-panel">
      {/* Baris atas: nama fase + info segmen */}
      <div className="sub-progress-header">
        <span className="sub-progress-phase">
          <span className="sub-progress-phase-icon">{phaseIcon}</span>
          {phase || 'Memproses...'}
        </span>
        {hasSegment && (
          <span className="sub-progress-segment">
            Segmen {segment} / {total_segments}
          </span>
        )}
      </div>

      {/* Progress bar dengan shimmer */}
      <div className="sub-progress-bar-bg">
        <div
          className="sub-progress-bar-fill"
          style={{ width: `${fillPct}%` }}
        />
      </div>

      {/* Label persentase + step */}
      <div className="sub-progress-step-row">
        <span className="sub-progress-pct">{fillPct.toFixed(1)}%</span>
        {step != null && total_steps != null && (
          <span className="sub-progress-step-detail">
            langkah {step} / {total_steps}
          </span>
        )}
      </div>

      {/* Baris bawah: kecepatan + ETA */}
      <div className="sub-progress-footer">
        {hasSpeed ? (
          <span className="speed-badge">
            ⚡ {speed.toFixed(1)} it/s
          </span>
        ) : (
          <span />
        )}
        {etaText && (
          <span className="eta-label">ETA {etaText}</span>
        )}
      </div>
    </div>
  );
}

function ScrollingLogs({ logs }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div ref={containerRef} className="terminal-logs">
      {logs && logs.length > 0 ? (
        logs.map((log, index) => {
          let logClass = "system";
          if (log.toLowerCase().includes("completed") || log.toLowerCase().includes("success")) {
            logClass = "success";
          } else if (log.toLowerCase().includes("failed") || log.toLowerCase().includes("error")) {
            logClass = "error";
          } else if (log.toLowerCase().includes("warning")) {
            logClass = "warning";
          }
          
          return (
            <div key={index} className={`log-line ${logClass}`}>
              {log}
            </div>
          );
        })
      ) : (
        <div className="log-line system">[SYSTEM] Establishing log connection stream...</div>
      )}
    </div>
  );
}

export default function TasksList({ tasks, onCancelTask, onDeleteTask, onResumeTask }) {
  const [expandedTaskLogs, setExpandedTaskLogs] = useState({});

  const toggleLogs = (taskId) => {
    setExpandedTaskLogs(prev => ({
      ...prev,
      [taskId]: !prev[taskId]
    }));
  };

  const activeTasks = Object.values(tasks);

  return (
    <div className="tab-content fade-in">
      <div className="glass-card">
        <h2 className="section-title" style={{ justifyContent: 'space-between' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Terminal size={20} /> Active Render Tasks ({activeTasks.length})
          </span>
          {activeTasks.some(t => t.status === 'processing') && (
            <Loader2 size={16} className="spin-loader" style={{ color: 'var(--color-primary)' }} />
          )}
        </h2>

        <div className="tasks-container">
          {activeTasks.length === 0 ? (
            <div className="no-tasks">
              <p>No active tasks in queue.</p>
              <p style={{ fontSize: '0.8rem', marginTop: '6px' }}>Go to the Video Generator tab to compile a new video.</p>
            </div>
          ) : (
            activeTasks.map(task => {
              const isExpanded = expandedTaskLogs[task.task_id];
              const statusLower = task.status ? task.status.toLowerCase() : '';
              const isProcessing = statusLower === 'processing';
              const isCompleted = statusLower === 'completed';
              const isCancelled = statusLower === 'cancelled';
              const isFailed = statusLower === 'failed';
              const canResume = isCancelled || isFailed;
              const canDelete = !isProcessing;
              
              return (
                <div key={task.task_id} className="task-card">
                  <div className="task-card-header">
                    <div className="task-info-top">
                      <h4>{task.video_subject ? task.video_subject.split('.')[0] : "AI Video compilation"}</h4>
                      <span className="task-id">ID: {task.task_id}</span>
                    </div>

                    <span className={`task-status-badge ${task.status.toLowerCase()}`}>
                      {task.status}
                    </span>
                  </div>

                  <div className="progress-section">
                    <div className="progress-info">
                      <span>Step: <b>{task.step || 'Queueing'}</b></span>
                      <span>{task.progress}%</span>
                    </div>
                    <div className="progress-bar-bg">
                      <div 
                        className="progress-bar-fill" 
                        style={{ 
                          width: `${task.progress}%`,
                          background: isCompleted 
                            ? 'var(--color-success)' 
                            : 'linear-gradient(to right, var(--color-primary), var(--color-secondary))'
                        }} 
                      />
                    </div>
                  </div>

                  {/* Panel sub-progress — hanya muncul saat task sedang berjalan */}
                  {isProcessing && task.sub_progress && (
                    <SubProgressPanel subProgress={task.sub_progress} />
                  )}

                  <div className="task-actions">
                    <button 
                      className="logs-toggle-btn"
                      onClick={() => toggleLogs(task.task_id)}
                    >
                      {isExpanded ? 'Hide Diagnostics Console' : 'Show Diagnostics Console'}
                    </button>

                    <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
                      {isProcessing && onCancelTask && (
                        <button 
                          className="btn btn-danger btn-icon-only"
                          onClick={() => onCancelTask(task.task_id)}
                          title="Cancel task"
                        >
                          <XCircle size={16} />
                        </button>
                      )}

                      {canResume && onResumeTask && (
                        <button 
                          className="btn btn-secondary btn-icon-only"
                          onClick={() => onResumeTask(task.task_id)}
                          title="Resume task"
                          style={{ color: 'var(--color-primary)' }}
                        >
                          <RefreshCw size={16} />
                        </button>
                      )}

                      {canDelete && onDeleteTask && (
                        <button 
                          className="btn btn-danger btn-icon-only"
                          onClick={() => {
                            if (window.confirm("Are you sure you want to delete this task? All associated files will be removed.")) {
                              onDeleteTask(task.task_id);
                            }
                          }}
                          title="Delete task"
                        >
                          <Trash2 size={16} />
                        </button>
                      )}
                    </div>
                  </div>

                  {isExpanded && (
                    <ScrollingLogs logs={task.logs} />
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
