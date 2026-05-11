import { useState, useRef, useEffect } from 'react';
import type { ModelType } from 'demucs-web';

export interface ModelInfo {
    id: ModelType;
    name: string;
    stems: number;
    sizeMB: number;
    badge?: 'recommended' | 'experimental';
}

export const MODELS: ModelInfo[] = [
    { id: 'htdemucs', name: 'Demucs v4', stems: 4, sizeMB: 161, badge: 'recommended' },
    { id: 'htdemucs_6s', name: 'Demucs v4 6-stem', stems: 6, sizeMB: 105 },
];

interface ModelPickerProps {
    selectedModel: ModelType;
    modelLoaded: boolean;
    modelLoading: boolean;
    onModelSelect: (model: ModelType) => void;
}

export function ModelPicker({
    selectedModel,
    modelLoaded,
    modelLoading,
    onModelSelect,
}: ModelPickerProps) {
    const [showMenu, setShowMenu] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);

    const currentModel = MODELS.find(m => m.id === selectedModel)!;

    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setShowMenu(false);
            }
        };
        if (showMenu) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [showMenu]);

    const handleSelect = (model: ModelType) => {
        setShowMenu(false);
        onModelSelect(model);
    };

    return (
        <div className="model-dropdown" ref={dropdownRef}>
            <button
                onClick={() => !modelLoading && setShowMenu(!showMenu)}
                className="model-dropdown-btn"
                disabled={modelLoading}
            >
                {/* Status indicator */}
                <div 
                    className="w-2.5 h-2.5 rounded-full"
                    style={{
                        background: modelLoading 
                            ? '#666' 
                            : modelLoaded 
                                ? '#22c55e' 
                                : '#444',
                        boxShadow: modelLoaded ? '0 0 8px rgba(34, 197, 94, 0.5)' : 'none'
                    }}
                />

                <div className="text-left">
                    <div className="text-sm font-medium text-white">
                        {modelLoading ? 'Loading...' : modelLoaded ? currentModel.name : 'Select Model'}
                    </div>
                    {!modelLoading && (
                        <div className="text-xs text-[#666]">
                            {modelLoaded ? `${currentModel.stems} stems • ${currentModel.sizeMB}MB` : 'Choose to start'}
                        </div>
                    )}
                </div>

                {!modelLoading && (
                    <svg 
                        className={`w-4 h-4 text-[#666] ml-2 transition-transform ${showMenu ? 'rotate-180' : ''}`}
                        viewBox="0 0 24 24" 
                        fill="currentColor"
                    >
                        <path d="M7 10l5 5 5-5H7z" />
                    </svg>
                )}

                {modelLoading && (
                    <svg className="w-4 h-4 text-[#666] ml-2 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                )}
            </button>

            {showMenu && (
                <div className="model-dropdown-menu">
                    {MODELS.map((model) => {
                        const isLoaded = modelLoaded && selectedModel === model.id;
                        return (
                            <div
                                key={model.id}
                                className={`model-dropdown-item ${isLoaded ? 'selected' : ''}`}
                                onClick={() => handleSelect(model.id)}
                            >
                                <div 
                                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                                    style={{
                                        background: isLoaded ? '#22c55e' : '#333',
                                        boxShadow: isLoaded ? '0 0 8px rgba(34, 197, 94, 0.5)' : 'none'
                                    }}
                                />

                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="font-medium text-white">{model.name}</span>
                                        {model.badge && (
                                            <span className={`status-badge ${model.badge}`}>
                                                {model.badge === 'recommended' ? 'Best' : 'Beta'}
                                            </span>
                                        )}
                                    </div>
                                    <div className="text-xs text-[#666] mt-0.5">
                                        {model.stems} stems • {model.sizeMB} MB download
                                    </div>
                                </div>

                                {isLoaded && (
                                    <svg className="w-4 h-4 text-[#22c55e] flex-shrink-0" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
                                    </svg>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
