import { Outlet } from 'react-router-dom';
import { useCallback, useMemo, useRef } from 'react';
import Header from './ui/Header';
import { HomeResetContext, type HomeHandlers } from './home-reset';

export function Layout() {
    // Home registers its handlers here; the header calls them. A ref keeps
    // the wrappers stable while always calling Home's latest handlers.
    const homeRef = useRef<HomeHandlers | null>(null);
    const register = useCallback((handlers: HomeHandlers | null) => {
        homeRef.current = handlers;
    }, []);
    const trigger = useCallback(() => {
        homeRef.current?.reset();
    }, []);
    const confirmLeave = useCallback(() => homeRef.current?.confirmLeave() ?? true, []);
    const control = useMemo(
        () => ({ register, trigger, confirmLeave }),
        [register, trigger, confirmLeave],
    );

    return (
        <HomeResetContext.Provider value={control}>
            <div className="stage">
                <div className="grain" />
                <Header />
                <main className="flex-1 flex flex-col">
                    <Outlet />
                </main>
            </div>
        </HomeResetContext.Provider>
    );
}
