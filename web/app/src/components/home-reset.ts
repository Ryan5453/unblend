import { createContext, useContext } from 'react';

/**
 * Lets the site chrome (the header wordmark and the "Studio" nav link) trigger
 * the Home page's soft reset — the same thing the "New File" button does —
 * without remounting Home (which would needlessly reload the model).
 *
 * Home registers its reset handler via `register`; the header calls `trigger`.
 */
export type ResetFn = () => void;

export interface HomeResetControl {
    register: (fn: ResetFn | null) => void;
    trigger: () => void;
}

export const HomeResetContext = createContext<HomeResetControl | null>(null);

export function useHomeReset(): HomeResetControl | null {
    return useContext(HomeResetContext);
}
