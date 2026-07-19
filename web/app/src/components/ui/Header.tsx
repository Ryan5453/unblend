import { Link, useLocation } from 'react-router-dom';

export default function Header() {
    const { pathname } = useLocation();

    return (
        <header className="site-header">
            <Link to="/" className="wm">un<i>/</i>blend</Link>
            <nav>
                <Link to="/" className={pathname === '/' ? 'on' : undefined}>Studio</Link>
                <Link to="/about" className={pathname === '/about' ? 'on' : undefined}>About</Link>
                <Link to="/privacy" className={pathname === '/privacy' ? 'on' : undefined}>Privacy</Link>
                <a href="https://github.com/Ryan5453/unblend" target="_blank" rel="noopener noreferrer">GitHub</a>
            </nav>
        </header>
    );
}
