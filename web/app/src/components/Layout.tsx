import { Outlet } from 'react-router-dom';
import { useCallback, useMemo, useRef } from 'react';
import Header from './ui/Header';
import { HomeResetContext, type ResetFn } from './home-reset';

export function Layout() {
    // Home registers its soft-reset here; the header triggers it. A ref keeps
    // the wrapper stable while always calling Home's latest handler.
    const resetRef = useRef<ResetFn | null>(null);
    const register = useCallback((fn: ResetFn | null) => {
        resetRef.current = fn;
    }, []);
    const trigger = useCallback(() => {
        resetRef.current?.();
    }, []);
    const control = useMemo(() => ({ register, trigger }), [register, trigger]);

    return (
        <HomeResetContext.Provider value={control}>
            <div className="stage">
                <div className="grain" />
                <Header />
                <div className="flex-1 flex flex-col">
                    <Outlet />
                </div>
            </div>
        </HomeResetContext.Provider>
    );
}
