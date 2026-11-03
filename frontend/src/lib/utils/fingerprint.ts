/**
 * Short fingerprint of a text (cyrb53: 53 bits written in hex). It tells two
 * values apart without copying them, for values that are already kept in
 * full elsewhere, such as the headers of an MCP server. It is not a
 * cryptographic hash.
 *
 * @param pText - The text to fingerprint
 * @returns The fingerprint in hex
 */
export function fFingerprint(pText: string): string {
  let vH1 = 0xdeadbeef;
  let vH2 = 0x41c6ce57;
  for (let vI = 0; vI < pText.length; vI++) {
    const cCode = pText.charCodeAt(vI);
    vH1 = Math.imul(vH1 ^ cCode, 2654435761);
    vH2 = Math.imul(vH2 ^ cCode, 1597334677);
  }
  vH1 = Math.imul(vH1 ^ (vH1 >>> 16), 2246822507);
  vH1 ^= Math.imul(vH2 ^ (vH2 >>> 13), 3266489909);
  vH2 = Math.imul(vH2 ^ (vH2 >>> 16), 2246822507);
  vH2 ^= Math.imul(vH1 ^ (vH1 >>> 13), 3266489909);

  return (4294967296 * (2097151 & vH2) + (vH1 >>> 0)).toString(16);
}
