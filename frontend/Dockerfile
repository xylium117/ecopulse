# EcoPulse frontend — static dashboard served by nginx
FROM nginx:1.27-alpine

# Copy nginx reverse proxy & caching configuration
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy all frontend assets (HTML, CSS, JS) to nginx public root
COPY frontend/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
