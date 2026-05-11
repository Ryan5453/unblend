import { useState, useRef, useEffect } from 'react';
import type { ModelType } from 'demucs-web';

export type ExecutionBackend = 'wasm' | 'webgpu';

export interface ModelInfo {
    id: ModelType;
    name: string;
    stems: number;
    sizeMB: number;
    supportsWebGPU: boolean;
    badge?: 'recommended' | 'experimental';
}

export const MODELS: ModelInfo[] = [
    { id: 'htdemucs', name: 'Demucs v4', stems: 4, sizeMB: 161, supportsWebGPU: true, badge: 'recommended' },
    { id: 'htdemucs_6s', name: 'Demucs v4 (6-stem)', stems: 6, sizeMB: 105, supportsWebGPU: true },
];

interface SettingsProps {
    selectedModel: ModelType;
    selectedBackend: ExecutionBackend;
    onModelChange: (model: ModelType) => void;
    onBackendChange: (backend: ExecutionBackend) => void;
}

export function Settings({
    selectedModel,
    selectedBackend,
    onModelChange,
    onBackendChange,
}: SettingsProps) {
    const [isOpen, setIsOpen] = useState(false);
    const popoverRef = useRef<HTMLDivElement>(null);

    const currentModel = MODELS.find(m => m.id === selectedModel)!;
    const webGPUAvailable = typeof navigator !== 'undefined' && 'gpu' in navigator;

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (popoverRef.current && !popoverRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };
        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isOpen]);

    // If model doesn't support WebGPU and WebGPU is selected, switch to WASM
    useEffect(() => {
        if (!currentModel.supportsWebGPU && selectedBackend === 'webgpu') {
            onBackendChange('wasm');
        }
    }, [selectedModel, selectedBackend, currentModel.supportsWebGPU, onBackendChange]);

    return (
        <div className="relative" ref={popoverRef}>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="settings-btn"
                title="Settings"
            >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
                </svg>
            </button>

            {isOpen && (
                <div className="settings-popover">
                    <div className="settings-section">
                        <div className="settings-label">Model</div>
                        <div className="settings-options">
                            {MODELS.map((model) => (
                                <button
                                    key={model.id}
                                    onClick={() => onModelChange(model.id)}
                                    className={`settings-option ${selectedModel === model.id ? 'active' : ''}`}
                                >
                                    <div className="flex items-center gap-2">
                                        <span>{model.name}</span>
                                        {model.badge && (
                                            <span className={`status-badge ${model.badge}`}>
                                                {model.badge === 'recommended' ? 'Best' : 'Beta'}
                                            </span>
                                        )}
                                    </div>
                                    <span className="settings-option-desc">
                                        {model.stems} stems • {model.sizeMB}MB
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="settings-section">
                        <div className="settings-label">Backend</div>
                        <div className="settings-options">
                            <button
                                onClick={() => onBackendChange('wasm')}
                                className={`settings-option ${selectedBackend === 'wasm' ? 'active' : ''}`}
                            >
                                <span>WebAssembly</span>
                                <span className="settings-option-desc">Slower, runs on CPU</span>
                            </button>
                            <button
                                onClick={() => currentModel.supportsWebGPU && webGPUAvailable && onBackendChange('webgpu')}
                                disabled={!currentModel.supportsWebGPU || !webGPUAvailable}
                                className={`settings-option ${selectedBackend === 'webgpu' ? 'active' : ''} ${!currentModel.supportsWebGPU || !webGPUAvailable ? 'disabled' : ''}`}
                            >
                                <span>WebGPU</span>
                                <span className="settings-option-desc">
                                    {!webGPUAvailable 
                                        ? 'Not supported in browser' 
                                        : !currentModel.supportsWebGPU 
                                            ? 'Not available for this model' 
                                            : 'Faster, GPU accelerated'}
                                </span>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
