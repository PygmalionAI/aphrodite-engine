import { defineConfig } from 'vitepress';

export default defineConfig({
  title: "Aphrodite Engine",
  description: "User and Developer Documentation",
  themeConfig: {
		// nav: [{ text: "Home", link: "/" }],

		sidebar: [
			{
				text: "Installation",
				link: "/pages/installation",
				items: [
					{
						text: "NVIDIA GPU",
						link: "/pages/installation/installation",
					},
					{
						text: "AMD ROCm",
						link: "/pages/installation/installation-rocm",
					},
					{
						text: "CPU",
						link: "/pages/installation/installation-cpu",
					},
					{
						text: "AWS Trainium1/Inferentia2",
						link: "/pages/installation/installation-neuron",
					},
					{
						text: "Google TPU",
						link: "/pages/installation/installation-tpu",
					},
					{
						text: "Intel XPU",
						link: "/pages/installation/installation-xpu",
					},
				],
			},
			{
				text: "Usage",
				link: "/pages/usage",
				items: [
					{
						text: "Quick Start",
						link: "/pages/usage/getting-started",
					},
					{
						text: "Debugging Instructions",
						link: "/pages/usage/debugging",
					}
				]
			}
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
