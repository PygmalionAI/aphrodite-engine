import { defineConfig } from 'vitepress';

export default defineConfig({
  title: "Aphrodite Engine",
  head: [['link', { rel: 'icon', href: '/favicon.ico' }]],
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
					},
					{
						text: "OpenAI API",
						link: "/pages/usage/openai",
					},
					{
						text: "Distributed Inference",
						link: "/pages/usage/distributed",
					},
					{
						text: "Production Metrics",
						link: "/pages/usage/metrics",
					},
					{
						text: "Supported Models",
						link: "/pages/usage/models",
					},
				]
			},
			{
				text: "Developer Documentation",
				link: "/pages/developer",
				items: [
					{
						text: "Adding a New Model",
						link: "/pages/developer/adding-model",
					},
					{
						text: "Ading a Multimodal Model",
						link: "/pages/developer/multimodal",
					},
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
