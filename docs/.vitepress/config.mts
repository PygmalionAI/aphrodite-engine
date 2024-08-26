import { defineConfig } from 'vitepress';
import { getSidebar } from "vitepress-plugin-auto-sidebar";

export default defineConfig({
  title: "Aphrodite Docs",
  description: "Documentation for the Aphrodite engine.",
  themeConfig: {
		// nav: [{ text: "Home", link: "/" }],

		sidebar: [
			...getSidebar({
				contentRoot: "/",

				// list of folders to include in the sidebar
				contentDirs: [
					"pages/api",
					"pages/other",
					"pages/markdown"
				],

				collapsible: true,
				collapsed: false,
			}),
		],

		socialLinks: [
			{ icon: "github", link: "https://github.com/PygmalionAI/aphrodite-engine" },
		],

		search: {
			provider: "local",
			options: {
				detailedView: true,
			},
		},
  }
})
