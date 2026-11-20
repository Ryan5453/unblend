import { Link, useLocation } from 'react-router-dom';

export function NotFound() {
    const { pathname } = useLocation();
    return (
        <div className="content-page">
            <h1 className="content-title">Page not found</h1>

            <div className="content-body">
                <p>
                    There is nothing at{' '}
                    <span className="not-found-path">
                        <strong>{pathname}</strong>.
                    </span>
                </p>
                <p>
                    <Link to="/">Back to the studio</Link>
                </p>
            </div>
        </div>
    );
}
