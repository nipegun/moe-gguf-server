import { cRoutes } from '$lib/constants/routes';

export class RouterService {
  static mChat(pId: string): string {
    return `${cRoutes.CHAT}/${pId}`;
  }

  static mSettings(pSection: string): string {
    return `${cRoutes.SETTINGS}/${pSection}`;
  }
}
