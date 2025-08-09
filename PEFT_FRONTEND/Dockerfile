# Stage 1: build React app
FROM node:22 AS build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Stage 2: serve with nginx
FROM nginx:alpine

# 把构建产物复制到 /usr/share/nginx/html/wardrobex 目录
COPY --from=build /app/dist /usr/share/nginx/html/wardrobex

# 拷贝自定义 nginx 配置文件
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
