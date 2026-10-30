import { createServer } from "node:http";
import { findProduct, products } from "./products.js";
import { page, productDetail, productList } from "./views.js";

export function createApp() {
  return createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");

    if (req.method === "GET" && url.pathname === "/") {
      return send(res, 200, page("Storefront", productList(products)));
    }

    const match = url.pathname.match(/^\/products\/([\w-]+)$/);
    if (req.method === "GET" && match) {
      const product = findProduct(match[1]);
      if (!product) return send(res, 404, page("Not found", "<p>No such product.</p>"));
      return send(res, 200, page(product.name, productDetail(product)));
    }

    send(res, 404, page("Not found", "<p>Page not found.</p>"));
  });
}

function send(res, status, html) {
  res.writeHead(status, { "Content-Type": "text/html; charset=utf-8" });
  res.end(html);
}
