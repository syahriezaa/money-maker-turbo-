import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Sparkles, 
  Terminal, 
  Film, 
  Sliders, 
  Coins, 
  Activity,
  Heart
} from 'lucide-react';
import Dashboard from './components/Dashboard';
import VideoGenerator from './components/VideoGenerator';
import TasksList from './components/TasksList';
import MediaLibrary from './components/MediaLibrary';
import Settings from './components/Settings';
import CustomPlayer from './components/CustomPlayer';
import { getVideos, createVideoTask, getTaskStatus, getTasks, cancelTask, deleteTask, resumeTask } from './api';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [tasks, setTasks] = useState({});
  const [videos, setVideos] = useState([]);
  const [isVideosLoading, setIsVideosLoading] = useState(false);
  const [playingVideo, setPlayingVideo] = useState(null);

  // Load videos
  const loadVideos = async () => {
    setIsVideosLoading(true);
    try {
      const data = await getVideos();
      // Sort newest first
      const sorted = [...data].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setVideos(sorted);
    } catch (err) {
      console.error("Failed to load videos:", err);
    } finally {
      setIsVideosLoading(false);
    }
  };

  // Initial load and active task resume check
  useEffect(() => {
    const loadAllTasksAndCheckActive = async () => {
      try {
        const allTasks = await getTasks();
        const taskMap = {};
        
        // Populate tasks map
        if (Array.isArray(allTasks)) {
          allTasks.forEach(task => {
            taskMap[task.task_id] = task;
          });
        } else if (typeof allTasks === 'object' && allTasks !== null) {
          // If returned as object map
          Object.assign(taskMap, allTasks);
        }
        
        setTasks(taskMap);
        
        const activeTaskId = localStorage.getItem("active_task_id");
        if (activeTaskId) {
          try {
            const activeTask = await getTaskStatus(activeTaskId);
            setTasks(prev => ({
              ...prev,
              [activeTaskId]: activeTask
            }));
            if (activeTask.status === 'processing') {
              setActiveTab("tasks");
            } else {
              localStorage.removeItem("active_task_id");
            }
          } catch (err) {
            console.error("Failed to query active task status:", err);
          }
        }
      } catch (err) {
        console.error("Failed to load tasks on mount:", err);
      }
    };
    loadAllTasksAndCheckActive();
    loadVideos();
  }, []);

  // Poll processing tasks
  useEffect(() => {
    const activeTaskIds = Object.keys(tasks).filter(
      id => tasks[id].status === 'processing'
    );
    
    if (activeTaskIds.length === 0) return;

    const interval = setInterval(async () => {
      let hasChanges = false;
      let hasCompletedTask = false;
      const updatedTasks = { ...tasks };

      for (const id of activeTaskIds) {
        try {
          const status = await getTaskStatus(id);
          
          // Check if changed
          if (
            status.status !== updatedTasks[id].status ||
            status.progress !== updatedTasks[id].progress ||
            status.logs.length !== updatedTasks[id].logs.length ||
            status.step !== updatedTasks[id].step
          ) {
            updatedTasks[id] = status;
            hasChanges = true;
          }

          if (status.status === 'Completed') {
            hasCompletedTask = true;
            const storedId = localStorage.getItem("active_task_id");
            if (storedId === id) {
              localStorage.removeItem("active_task_id");
            }
          }
        } catch (err) {
          console.error(`Error polling task ${id}:`, err);
        }
      }

      if (hasChanges) {
        setTasks(updatedTasks);
      }

      if (hasCompletedTask) {
        loadVideos(); // refresh media gallery
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [tasks]);

  // Handle task queue submission
  const handleQueueTask = async (taskPayload) => {
    const taskStatus = await createVideoTask(taskPayload);
    setTasks(prev => ({
      ...prev,
      [taskStatus.task_id]: taskStatus
    }));
    localStorage.setItem("active_task_id", taskStatus.task_id);
    setActiveTab("tasks"); // Switch to tasks list
  };

  const handleCancelTask = async (taskId) => {
    try {
      await cancelTask(taskId);
      const status = await getTaskStatus(taskId);
      setTasks(prev => ({
        ...prev,
        [taskId]: status
      }));
    } catch (err) {
      console.error("Failed to cancel task:", err);
      alert("Failed to cancel task: " + err.message);
    }
  };

  const handleDeleteTask = async (taskId) => {
    try {
      await deleteTask(taskId);
      setTasks(prev => {
        const copy = { ...prev };
        delete copy[taskId];
        return copy;
      });
      loadVideos();
    } catch (err) {
      console.error("Failed to delete task:", err);
      alert("Failed to delete task: " + err.message);
    }
  };

  const handleResumeTask = async (taskId) => {
    try {
      await resumeTask(taskId);
      const status = await getTaskStatus(taskId);
      setTasks(prev => ({
        ...prev,
        [taskId]: status
      }));
    } catch (err) {
      console.error("Failed to resume task:", err);
      alert("Failed to resume task: " + err.message);
    }
  };

  // Count active tasks
  const activeTasksCount = Object.values(tasks).filter(t => t.status === 'processing').length;

  return (
    <div className="app-layout">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand">
          <Sparkles className="brand-icon" size={24} />
          <h1 className="brand-name">MoneyPrinter<span>Turbo</span></h1>
        </div>

        <nav className="sidebar-nav">
          <div 
            className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            <LayoutDashboard className="nav-icon" />
            <span>Dashboard</span>
          </div>

          <div 
            className={`nav-item ${activeTab === 'generator' ? 'active' : ''}`}
            onClick={() => setActiveTab('generator')}
          >
            <Sparkles className="nav-icon" />
            <span>Video Generator</span>
          </div>

          <div 
            className={`nav-item ${activeTab === 'tasks' ? 'active' : ''}`}
            onClick={() => setActiveTab('tasks')}
          >
            <Terminal className="nav-icon" />
            <span>Active Tasks</span>
            {activeTasksCount > 0 && (
              <span style={{ 
                marginLeft: 'auto', 
                background: 'var(--color-primary)', 
                color: 'white', 
                fontSize: '0.7rem', 
                fontWeight: 700, 
                padding: '2px 6px', 
                borderRadius: '8px' 
              }}>
                {activeTasksCount}
              </span>
            )}
          </div>

          <div 
            className={`nav-item ${activeTab === 'media' ? 'active' : ''}`}
            onClick={() => setActiveTab('media')}
          >
            <Film className="nav-icon" />
            <span>Media Library</span>
            {videos.length > 0 && (
              <span style={{ 
                marginLeft: 'auto', 
                background: 'rgba(255,255,255,0.06)', 
                color: 'var(--text-secondary)', 
                fontSize: '0.7rem', 
                fontWeight: 700, 
                padding: '2px 6px', 
                borderRadius: '8px' 
              }}>
                {videos.length}
              </span>
            )}
          </div>

          <div 
            className={`nav-item ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            <Sliders className="nav-icon" />
            <span>Settings</span>
          </div>
        </nav>

        <div className="sidebar-footer">
          <div className="user-profile">
            <div className="user-avatar">AD</div>
            <div className="user-info">
              <span className="user-name">Admin Creator</span>
              <span className="user-role">Super Account</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Section container */}
      <main className="main-content">
        <header className="content-header">
          <div className="header-title">
            <h1>
              {activeTab === 'dashboard' && 'Control Panel'}
              {activeTab === 'generator' && 'Video Composer'}
              {activeTab === 'tasks' && 'Render Operations'}
              {activeTab === 'media' && 'Media Library'}
              {activeTab === 'settings' && 'System Parameters'}
            </h1>
            <p>
              {activeTab === 'dashboard' && 'Automated content creation metrics & pipeline status.'}
              {activeTab === 'generator' && 'Draft script narration and compile synthetic videos.'}
              {activeTab === 'tasks' && 'Track progress timelines and execution diagnostics console.'}
              {activeTab === 'media' && 'Browse, play and download your generated videos.'}
              {activeTab === 'settings' && 'Configure API integrations, voice engines, and output rules.'}
            </p>
          </div>
          
          <div className="header-actions">
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px', 
              background: 'rgba(255,255,255,0.03)', 
              border: '1px solid var(--border-color)', 
              borderRadius: '12px', 
              padding: '8px 16px',
              fontSize: '0.85rem'
            }}>
              <Coins size={14} style={{ color: 'var(--color-accent)' }} />
              <span>Balance: <b>$85.72</b></span>
            </div>
          </div>
        </header>

        {/* Tab Render Switcher */}
        {activeTab === 'dashboard' && (
          <Dashboard 
            stats={{}} 
            activeTasksCount={activeTasksCount}
            videosCount={videos.length}
            onNavigate={setActiveTab}
          />
        )}
        
        {activeTab === 'generator' && (
          <VideoGenerator onSubmitTask={handleQueueTask} />
        )}

        {activeTab === 'tasks' && (
          <TasksList 
            tasks={tasks} 
            onCancelTask={handleCancelTask}
            onDeleteTask={handleDeleteTask}
            onResumeTask={handleResumeTask}
          />
        )}

        {activeTab === 'media' && (
          <MediaLibrary 
            videos={videos} 
            isLoading={isVideosLoading} 
            onPlayVideo={setPlayingVideo} 
          />
        )}

        {activeTab === 'settings' && (
          <Settings />
        )}
      </main>

      {/* Custom Video Player Modal Overlay */}
      {playingVideo && (
        <CustomPlayer 
          video={playingVideo} 
          onClose={() => setPlayingVideo(null)} 
        />
      )}
    </div>
  );
}
