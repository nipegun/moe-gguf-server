/**
 *
 * NAVIGATION & MENUS
 *
 * Components for dropdown menus and action selection.
 *
 */

/**
 * **DropdownMenuSearchable** - Searchable content for dropdown menus
 *
 * Renders a search input with filtered content area, empty state, and optional footer.
 * Designed to be injected into any dropdown container (DropdownMenu.Content,
 * DropdownMenu.SubContent, etc.) without providing its own Root.
 *
 * **Features:**
 * - Search/filter input
 * - Keyboard navigation support
 * - Custom content and footer via snippets
 * - Empty state message
 *
 * @example
 * ```svelte
 * <DropdownMenu.Root>
 *   <DropdownMenu.Trigger>...</DropdownMenu.Trigger>
 *   <DropdownMenu.Content class="pt-0">
 *     <DropdownMenuSearchable
 *       bind:searchValue
 *       placeholder="Search..."
 *       isEmpty={filteredItems.length === 0}
 *     >
 *       {#each items as item}<Item {item} />{/each}
 *     </DropdownMenuSearchable>
 *   </DropdownMenu.Content>
 * </DropdownMenu.Root>
 * ```
 */
export { default as DropdownMenuSearchable } from './DropdownMenuSearchable.svelte';

/**
 * **DropdownMenuActions** - Multi-action dropdown menu
 *
 * Dropdown menu for multiple action options with icons and shortcuts.
 * Supports destructive variants and keyboard shortcut hints.
 *
 * **Features:**
 * - Configurable trigger icon with tooltip
 * - Action items with icons and labels
 * - Destructive variant styling
 * - Keyboard shortcut display
 * - Separator support between groups
 *
 * @example
 * ```svelte
 * <DropdownMenuActions
 *   triggerIcon={MoreHorizontal}
 *   triggerTooltip="More actions"
 *   actions={[
 *     { icon: Edit, label: 'Edit', onclick: handleEdit },
 *     { icon: Trash, label: 'Delete', onclick: handleDelete, variant: 'destructive' }
 *   ]}
 * />
 * ```
 */
export { default as DropdownMenuActions } from './DropdownMenuActions.svelte';
/**
 * **SidebarNavigation** - Sidebar with actions menu and conversation list
 *
 * Collapsible sidebar displaying conversation history with search and
 * management actions. Integrates with ShadCN sidebar component for
 * consistent styling and mobile responsiveness.
 *
 * **Architecture:**
 * - Uses ShadCN Sidebar.* components for structure
 * - Fetches conversations from conversationsStore
 * - Hosts the global search field on the line of the logo
 * - Handles conversation CRUD operations via conversationsStore
 *
 * **Navigation:**
 * - Click conversation to navigate to `/chat/[id]`
 * - New chat button navigates to `/` (root)
 * - Active conversation highlighted based on route params
 *
 * **Conversation Management:**
 * - Right-click or menu button for context menu
 * - Rename: Opens inline edit dialog
 * - Delete: Shows confirmation with conversation preview
 * - Delete All: Removes all conversations with confirmation
 *
 * **Features:**
 * - Menu: New chat, Monitor, Model and Settings
 * - Global search (Cmd/Ctrl+K focuses it): chats by title and by the content of their messages,
 *   pages and their parts, settings, models, tools and MCP servers; a result opens its page or
 *   chat and highlights what it found ({@link SidebarNavigationGlobalSearch})
 * - Conversation list with message previews (first message truncated)
 * - Active conversation highlighting
 * - Mobile-responsive collapse/expand via ShadCN sidebar
 *
 * @example
 * ```svelte
 * <SidebarNavigation bind:this={sidebarRef} />
 * ```
 */
export { default as SidebarNavigation } from './SidebarNavigation/SidebarNavigation.svelte';

/**
 * Single conversation item in sidebar. Displays conversation title (truncated),
 * last message preview, and timestamp. Shows context menu on right-click with
 * rename and delete options. Highlights when active (matches current route).
 * Handles click to navigate and keyboard accessibility.
 */
export { default as SidebarNavigationConversationItem } from './SidebarNavigation/SidebarNavigationConversationItem.svelte';

/**
 * **SidebarNavigationSelectionBar** - Bulk action toolbar for selection mode
 *
 * Rendered above the conversation list when the sidebar enters selection mode.
 * Hosts a master checkbox (with select-all / clear-all semantics over the
 * currently-visible items), a selected-count caption, and bulk actions for
 * pin/unpin, export, and delete. Delete uses
 * {@link DialogConfirmation} before invoking the bulk store method.
 *
 * Pure-presentational; all operations are delegated via callbacks so the
 * sidebar owns selection state and persistence.
 *
 * @example
 * ```svelte
 * <SidebarNavigationSelectionBar
 *   selectedCount={selectedIds.size}
 *   visibleCount={visibleConversations.length}
 *   allVisibleSelected={...}
 *   someVisibleSelected={...}
 *   someSelectedPinned={...}
 *   onSelectAllToggle={toggleSelectAll}
 *   onBulkPinToggle={handleBulkPin}
 *   onBulkExport={handleBulkExport}
 *   onBulkDelete={handleBulkDelete}
 *   onClose={exitSelectionMode}
 * />
 * ```
 */
export { default as SidebarNavigationSelectionBar } from './SidebarNavigation/SidebarNavigationSelectionBar.svelte';

/**
 * **SidebarNavigationConversationList** - Grouped conversation list
 *
 * Pure-presentational list of conversations. Splits items into a Pinned
 * section and a Recent Conversations section with the unpinned items. Item
 * selection, edit, delete, and stop-generation are delegated to the caller
 * via callbacks.
 *
 * @example
 * ```svelte
 * <SidebarNavigationConversationList
 *   {filteredConversations}
 *   {currentChatId}
 *   onSelect={...}
 *   onEdit={...}
 *   onDelete={...}
 *   onStop={...}
 * />
 * ```
 */
export { default as SidebarNavigationConversationList } from './SidebarNavigation/SidebarNavigationConversationList.svelte';
export { default as SidebarNavigationActions } from './SidebarNavigation/SidebarNavigationActions.svelte';

/**
 * **SidebarNavigationGlobalSearch** - Results of the global search of the sidebar
 *
 * Groups what holds every word of the query (without regard to case or accents):
 * sections (pages, their parts and the sections of the settings), settings, models,
 * tools, MCP servers, chats by title (with their actions) and chats by the content of
 * their messages (read from IndexedDB a moment after the last key press). Opening a
 * result navigates to its page or chat and highlights the element it points to
 * (`fRevealSearchTarget`); `mOpenFirst()` opens the first one, for Enter.
 */
export { default as SidebarNavigationGlobalSearch } from './SidebarNavigation/SidebarNavigationGlobalSearch.svelte';
