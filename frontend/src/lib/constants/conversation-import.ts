// First bytes of every ZIP local file header ("PK"). Import detects an archive
// from these bytes rather than from the filename, which the OS may not preserve.
export const aZipMagic = [0x50, 0x4b];

// Import budgets. The archive is read and expanded in memory on the UI thread,
// so the selected file, the number of ZIP entries and the expanded size of the
// accepted `.jsonl` entries are all bounded before anything is decompressed.
export const cBytesPerMegabyte = 1024 * 1024;

/** Largest file (JSONL or ZIP) the import reads */
export const cImportFileMaxBytes = 512 * cBytesPerMegabyte;

/** Largest single JSONL document, plain or extracted from an archive */
export const cImportJsonlMaxBytes = 256 * cBytesPerMegabyte;

/** Most entries (of any kind) a ZIP archive may list */
export const cImportZipMaxEntries = 10000;

/** Largest combined extracted size of the `.jsonl` entries of a ZIP archive */
export const cImportZipMaxExtractedBytes = 1024 * cBytesPerMegabyte;
