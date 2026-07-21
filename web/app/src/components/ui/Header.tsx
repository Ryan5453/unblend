import { Link, useLocation } from 'react-router-dom';
import { useHomeReset } from '../home-reset';

export default function Header() {
    const { pathname } = useLocation();
    const reset = useHomeReset();

    // Clicking the wordmark or "Studio" should reset the app to a clean drop
    // screen (same as "New File"). When Home is mounted (on "/") this fires its
    // registered reset; when arriving from another page the Link navigation
    // mounts a fresh Home anyway, so the no-op here is harmless.
    const resetHome = () => reset?.trigger();

    return (
        <header className="site-header">
            <Link to="/" className="wm" onClick={resetHome}>un<i>/</i>blend</Link>
            <nav>
                <Link to="/" className={pathname === '/' ? 'on' : undefined} onClick={resetHome}>Studio</Link>
                <Link to="/about" className={pathname === '/about' ? 'on' : undefined}>About</Link>
                <Link to="/privacy" className={pathname === '/privacy' ? 'on' : undefined}>Privacy</Link>
                <a href="https://github.com/Ryan5453/unblend" target="_blank" rel="noopener noreferrer">GitHub</a>
            </nav>
        </header>
    );
}
