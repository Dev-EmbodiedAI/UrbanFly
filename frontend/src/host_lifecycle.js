// WebView2 can keep document.hidden=false while its WPF window is minimized.
// Host visibility supplements Page Visibility for presentation work only.
export const hostLifecycle = { hidden: false };

export function bindHostLifecycle() {
  window.chrome?.webview?.addEventListener('message', (event) => {
    if (event.data?.type === 'host_visibility') {
      hostLifecycle.hidden = event.data.hidden === true;
    }
  });
}

export const presentationHidden = () => document.hidden || hostLifecycle.hidden;
