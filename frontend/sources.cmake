# Inputs used to decide whether the npm build output is up-to-date.

set(UI_SOURCE_GLOBS
    src/*
    static/*
    scripts/*
)

set(UI_SOURCE_FILES
    package.json
    package-lock.json
    vite.config.ts
    svelte.config.js
    tsconfig.json
    pwa-assets.config.ts
    pwa-assets-dark.config.ts
)
