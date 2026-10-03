import { createContext, useContext } from 'react';

/**
 * Lets the site chrome (the header wordmark and nav links) reach the Home
 * page without remounting it (which would needlessly reload the model):
 * `trigger` runs Home's soft reset, the same thing the "New File" button does,
 * and `confirmLeave` asks before navigating away would discard its work.
 *
 * Home registers its handlers via `register`; the header calls the rest.
 */
export interface HomeHandlers {
    reset: () => void;
    /** Returns false if the user chose to stay on Home. */
    confirmLeave: () => boolean;
}

export interface HomeResetControl {
    register: (handlers: HomeHandlers | null) => void;
    trigger: () => void;
    confirmLeave: () => boolean;
}

export const HomeResetContext = createContext<HomeResetControl | null>(null);

export function useHomeReset(): HomeResetControl | null {
    return useContext(HomeResetContext);
}
