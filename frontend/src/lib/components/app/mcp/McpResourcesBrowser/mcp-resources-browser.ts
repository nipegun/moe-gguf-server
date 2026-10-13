import type { MCPResource, MCPResourceInfo } from '$lib/types';
import { parseResourcePath } from '$lib/utils';

export interface ResourceTreeNode {
  name: string;
  resource?: MCPResourceInfo;
  children: Map<string, ResourceTreeNode>;
  isFiltered?: boolean;
}

function fResourceMatchesSearch(pResource: MCPResource, pQuery: string): boolean {
  return (
    pResource.title?.toLowerCase().includes(pQuery) || pResource.uri.toLowerCase().includes(pQuery)
  );
}

export function fBuildResourceTree(
  pResourceList: MCPResource[],
  pServerName: string,
  pSearchQuery?: string
): ResourceTreeNode {
  const dRoot: ResourceTreeNode = { name: 'root', children: new Map() };

  if (!pSearchQuery || !pSearchQuery.trim()) {
    for (const cResource of pResourceList) {
      const cPathParts = parseResourcePath(cResource.uri);
      let vCurrent = dRoot;

      for (let vI = 0; vI < cPathParts.length - 1; vI++) {
        const cPart = cPathParts[vI];
        if (!vCurrent.children.has(cPart)) {
          vCurrent.children.set(cPart, { name: cPart, children: new Map() });
        }
        vCurrent = vCurrent.children.get(cPart)!;
      }

      const cFileName = cPathParts[cPathParts.length - 1] || cResource.name;
      vCurrent.children.set(cResource.uri, {
        name: cFileName,
        resource: { ...cResource, serverName: pServerName },
        children: new Map()
      });
    }

    return dRoot;
  }

  const cQuery = pSearchQuery.toLowerCase();

  // Build tree with filtering
  for (const cResource of pResourceList) {
    if (!fResourceMatchesSearch(cResource, cQuery)) continue;

    const cPathParts = parseResourcePath(cResource.uri);
    let vCurrent = dRoot;

    for (let vI = 0; vI < cPathParts.length - 1; vI++) {
      const cPart = cPathParts[vI];
      if (!vCurrent.children.has(cPart)) {
        vCurrent.children.set(cPart, { name: cPart, children: new Map(), isFiltered: true });
      }
      vCurrent = vCurrent.children.get(cPart)!;
    }

    const cFileName = cPathParts[cPathParts.length - 1] || cResource.name;

    vCurrent.children.set(cResource.uri, {
      name: cFileName,
      resource: { ...cResource, serverName: pServerName },
      children: new Map(),
      isFiltered: true
    });
  }

  function fCleanupEmptyFolders(pNode: ResourceTreeNode): boolean {
    if (pNode.resource) return true;

    const aToDelete: string[] = [];
    for (const [cName, cChild] of pNode.children.entries()) {
      if (!fCleanupEmptyFolders(cChild)) {
        aToDelete.push(cName);
      }
    }

    for (const cName of aToDelete) {
      pNode.children.delete(cName);
    }

    return pNode.children.size > 0;
  }

  fCleanupEmptyFolders(dRoot);

  return dRoot;
}

export function fCountTreeResources(pNode: ResourceTreeNode): number {
  if (pNode.resource) return 1;
  let vCount = 0;

  for (const cChild of pNode.children.values()) {
    vCount += fCountTreeResources(cChild);
  }

  return vCount;
}

export function fSortTreeChildren(pChildren: ResourceTreeNode[]): ResourceTreeNode[] {
  return pChildren.sort((pA, pB) => {
    const aIsFolder = !pA.resource && pA.children.size > 0;
    const cBIsFolder = !pB.resource && pB.children.size > 0;

    if (aIsFolder && !cBIsFolder) return -1;
    if (!aIsFolder && cBIsFolder) return 1;

    return pA.name.localeCompare(pB.name);
  });
}
