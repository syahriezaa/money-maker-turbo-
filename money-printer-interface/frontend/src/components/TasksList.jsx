import React, { useState, useEffect, useRef } from 'react';
import { Terminal, RefreshCw, XCircle, AlertCircle, PlayCircle, Loader2, Trash2 } from 'lucide-react';

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
