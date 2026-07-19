import { Outlet } from 'react-router-dom';
import Header from './ui/Header';

export function Layout() {
    return (
        <div className="stage">
            <div className="grain" />
            <Header />
            <div className="flex-1 flex flex-col">
                <Outlet />
            </div>
        </div>
    );
}
