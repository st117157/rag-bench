# Проверка LLM-судьи

Двадцать вердиктов, отобранных так, чтобы поровну попали и «верно», и «неверно»,
и отдельно — случаи, где ответ признан верным при проваленном поиске.

**Как заполнять.** Читаете вопрос, эталон и ответ системы, ставите свой вердикт:
`+` если ответ по смыслу совпадает с эталоном, `-` если нет. Вердикт судьи спрятан
под раскрывающимся блоком — не подглядывайте до того, как решили сами.

Считается доля совпадений. Это число идёт в README: без него колонка `correct`
ничего не стоит.

---

## 1. `q003` · подход `naive` · тип lexical

**Вопрос.** How do I set the status code of a path operation to 201?

*по-русски: Как задать код ответа 201 для операции пути?*

**Эталон.** Pass status_code to the path operation decorator; the status module provides named constants such as HTTP_201_CREATED.

**Ответ системы.**

```
NO DATA
```

Нужный документ в выдаче: **да**
Найдено: how-to__general.md, tutorial__response-status-code.md, deployment__docker.md
Ожидалось: tutorial__response-status-code.md

<details><summary>вердикт судьи</summary>

**НЕВЕРНО**
</details>

Мой вердикт: `-`

---

## 2. `q028` · подход `longctx_filtered` · тип semantic

**Вопрос.** One machine has many cores but my app only loads one of them. How do I use the rest?

*по-русски: На машине много ядер, но приложение грузит только одно. Как задействовать остальные?*

**Эталон.** Run several worker processes, for example with the workers option of the server or by starting multiple containers behind a load balancer.

**Ответ системы.**

```
To utilize multiple cores in your FastAPI application, you can run several worker processes simultaneously. This can be achieved by using a process manager like Uvicorn with the `--workers` option, or by deploying your application in a container orchestration system like Kubernetes, which can manage multiple instances of your application across the available cores. [source: deployment__concepts.md]
```

Нужный документ в выдаче: **НЕТ**
Найдено: deployment__concepts.md, deployment__docker.md, alternatives.md
Ожидалось: deployment__server-workers.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 3. `q033` · подход `reranked` · тип multihop

**Вопрос.** How do I accept an uploaded file together with ordinary form fields in one request?

*по-русски: Как принять в одном запросе загружаемый файл вместе с обычными полями формы?*

**Эталон.** Declare File and Form parameters in the same function; the body arrives as multipart form data, which requires python-multipart to be installed.

**Ответ системы.**

```
You can accept an uploaded file together with ordinary form fields in one request by using `File` and `Form` parameters together. This allows you to receive both data and files in the same request. [source: tutorial__request-forms-and-files.md]
```

Нужный документ в выдаче: **да**
Найдено: tutorial__request-forms-and-files.md, tutorial__request-files.md
Ожидалось: tutorial__request-files.md, tutorial__request-forms-and-files.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 4. `q018` · подход `longctx_filtered` · тип semantic

**Вопрос.** How do I answer the caller at once and let the lengthy part finish later?

*по-русски: Как ответить клиенту сразу, а долгую часть доделать потом?*

**Эталон.** Declare a BackgroundTasks parameter and register the function with add_task; it runs after the response has been returned.

**Ответ системы.**

```
You can use background tasks in FastAPI to handle lengthy operations after sending a response to the caller. This allows you to respond immediately while the lengthy task continues to run in the background. 

For more details, refer to the FastAPI documentation on background tasks. [source: advanced__events.md]
```

Нужный документ в выдаче: **НЕТ**
Найдено: advanced__events.md, async.md, help-fastapi.md
Ожидалось: tutorial__background-tasks.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `-`   ← расхождение с судьёй

---

## 5. `q002` · подход `longctx` · тип lexical

**Вопрос.** What does the response_model parameter do?

*по-русски: Что делает параметр response_model?*

**Эталон.** It declares the Pydantic model of the response, so FastAPI validates, filters and documents the returned data according to it.

**Ответ системы.**

```
The `response_model` parameter in FastAPI is used to specify a Pydantic model that defines the structure of the response data. FastAPI will use this model to serialize the data to JSON, ensuring that the response adheres to the defined schema and automatically generating documentation for it in the OpenAPI schema. This helps in validating and documenting the response data effectively. 

[source: advanced__response-model.md]
```

Нужный документ в выдаче: **НЕТ**
Найдено: about__index.md, advanced__additional-responses.md, advanced__additional-status-codes.md
Ожидалось: tutorial__response-model.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 6. `q036` · подход `hybrid` · тип multihop

**Вопрос.** How do I restrict an endpoint by OAuth2 scopes and then test it with an overridden dependency?

*по-русски: Как ограничить эндпоинт по OAuth2-скоупам и протестировать его с подменённой зависимостью?*

**Эталон.** Require the scopes through Security with a scopes list and verify them in the dependency; in tests replace that dependency using app.dependency_overrides.

**Ответ системы.**

```
NO DATA
```

Нужный документ в выдаче: **да**
Найдено: advanced__security__oauth2-scopes.md
Ожидалось: advanced__security__oauth2-scopes.md, advanced__testing-dependencies.md

<details><summary>вердикт судьи</summary>

**НЕВЕРНО**
</details>

Мой вердикт: `-`

---

## 7. `q005` · подход `hybrid` · тип lexical

**Вопрос.** How do I disable convert_underscores for a Header parameter?

*по-русски: Как отключить convert_underscores у параметра Header?*

**Эталон.** Set convert_underscores=False on Header. The docs warn that some proxies and servers reject headers containing underscores.

**Ответ системы.**

```
To disable automatic conversion of underscores to hyphens for a Header parameter, set the parameter `convert_underscores` of `Header` to `False`. However, be aware that some HTTP proxies and servers disallow the usage of headers with underscores. [source: tutorial__header-params.md]
```

Нужный документ в выдаче: **да**
Найдено: tutorial__header-params.md, tutorial__header-param-models.md
Ожидалось: tutorial__header-params.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 8. `q029` · подход `hybrid_gated` · тип semantic

**Вопрос.** How do I change the title and release number displayed at the top of the generated docs?

*по-русски: Как поменять заголовок и номер выпуска в шапке сгенерированной документации?*

**Эталон.** Pass title, version and description to the FastAPI application; they end up in the OpenAPI schema and in the docs UI.

**Ответ системы.**

```
NO DATA
```

Нужный документ в выдаче: **да**
Найдено: how-to__general.md, tutorial__path-operation-configuration.md, deployment__versions.md
Ожидалось: tutorial__metadata.md

<details><summary>вердикт судьи</summary>

**НЕВЕРНО**
</details>

Мой вердикт: `-`

---

## 9. `q005` · подход `reranked` · тип lexical

**Вопрос.** How do I disable convert_underscores for a Header parameter?

*по-русски: Как отключить convert_underscores у параметра Header?*

**Эталон.** Set convert_underscores=False on Header. The docs warn that some proxies and servers reject headers containing underscores.

**Ответ системы.**

```
To disable automatic conversion of underscores to hyphens for a Header parameter, set the parameter `convert_underscores` of `Header` to `False`. However, be aware that some HTTP proxies and servers disallow the usage of headers with underscores. [source: tutorial__header-params.md]
```

Нужный документ в выдаче: **да**
Найдено: tutorial__header-params.md, tutorial__header-param-models.md
Ожидалось: tutorial__header-params.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 10. `q028` · подход `grep` · тип semantic

**Вопрос.** One machine has many cores but my app only loads one of them. How do I use the rest?

*по-русски: На машине много ядер, но приложение грузит только одно. Как задействовать остальные?*

**Эталон.** Run several worker processes, for example with the workers option of the server or by starting multiple containers behind a load balancer.

**Ответ системы.**

```
NO DATA
```

Нужный документ в выдаче: **НЕТ**
Найдено: advanced__events.md, advanced__stream-data.md, deployment__manually.md
Ожидалось: deployment__server-workers.md

<details><summary>вердикт судьи</summary>

**НЕВЕРНО**
</details>

Мой вердикт: `-`

---

## 11. `q009` · подход `naive` · тип lexical

**Вопрос.** How do I mark a path operation as deprecated?

*по-русски: Как пометить операцию пути как устаревшую?*

**Эталон.** Pass the deprecated parameter to the path operation decorator; the endpoint keeps working but is shown as deprecated in the interactive docs.

**Ответ системы.**

```
To mark a path operation as deprecated, pass the parameter `deprecated` to the path operation decorator. This will clearly mark it as deprecated in the interactive docs. [source: tutorial__path-operation-configuration.md]
```

Нужный документ в выдаче: **да**
Найдено: how-to__general.md, tutorial__path-operation-configuration.md, tutorial__dependencies__dependencies-in-path-operation-decorators.md
Ожидалось: tutorial__path-operation-configuration.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 12. `q021` · подход `longctx` · тип semantic

**Вопрос.** How do I verify my routes automatically without starting a live server?

*по-русски: Как автоматически проверять маршруты, не поднимая работающий сервер?*

**Эталон.** Use TestClient from fastapi.testclient: wrap the app in it and call the endpoints directly with normal pytest assertions.

**Ответ системы.**

```
NO DATA
```

Нужный документ в выдаче: **НЕТ**
Найдено: about__index.md, advanced__additional-responses.md, advanced__additional-status-codes.md
Ожидалось: tutorial__testing.md

<details><summary>вердикт судьи</summary>

**НЕВЕРНО**
</details>

Мой вердикт: `-`

---

## 13. `q010` · подход `reranked` · тип lexical

**Вопрос.** What is the lifespan parameter of the FastAPI app?

*по-русски: Что такое параметр lifespan у приложения FastAPI?*

**Эталон.** It takes an async context manager with a yield: code before the yield runs at startup, code after it runs at shutdown. The older on_event startup and shutdown handlers are deprecated.

**Ответ системы.**

```
The `lifespan` parameter of the FastAPI app is used to define the *startup* and *shutdown* logic using a context manager, allowing you to execute specific code before the application starts taking requests and when it is shutting down. If you provide a `lifespan` parameter, the `startup` and `shutdown` event handlers will not be called. [source: advanced__events.md]
```

Нужный документ в выдаче: **да**
Найдено: advanced__events.md, reference__index.md, advanced__dataclasses.md
Ожидалось: advanced__events.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 14. `q040` · подход `longctx` · тип multihop

**Вопрос.** How do I give every request its own database session when using SQLModel?

*по-русски: Как выдавать каждому запросу собственную сессию базы при использовании SQLModel?*

**Эталон.** Declare a dependency with yield that opens a SQLModel Session, yields it to the path operation and closes it afterwards, then depend on it in every route that touches the database.

**Ответ системы.**

```
To give every request its own database session when using SQLModel, you can create a dependency that provides a new session for each request. This can be done by defining a function that creates a session and then using it as a dependency in your path operations. You can refer to the SQLModel documentation for specific examples on how to implement this. 

For more details, check the SQLModel docs [here](https://sqlmodel.tiangolo.com/). [source: tutorial__testing-database.md]
```

Нужный документ в выдаче: **НЕТ**
Найдено: about__index.md, advanced__additional-responses.md, advanced__additional-status-codes.md
Ожидалось: tutorial__sql-databases.md, tutorial__dependencies__dependencies-with-yield.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `-`   ← расхождение с судьёй

---

## 15. `q031` · подход `naive` · тип multihop

**Вопрос.** How do I deploy behind a reverse proxy that strips a path prefix, while still serving HTTPS correctly?

*по-русски: Как развернуть приложение за прокси, срезающим префикс пути, сохранив корректный HTTPS?*

**Эталон.** Set root_path to the stripped prefix and enable handling of the forwarded headers; TLS itself is terminated on the proxy, with certificates usually issued by Let's Encrypt.

**Ответ системы.**

```
NO DATA
```

Нужный документ в выдаче: **да**
Найдено: deployment__https.md, advanced__behind-a-proxy.md
Ожидалось: advanced__behind-a-proxy.md, deployment__https.md

<details><summary>вердикт судьи</summary>

**НЕВЕРНО**
</details>

Мой вердикт: `-`

---

## 16. `q003` · подход `longctx` · тип lexical

**Вопрос.** How do I set the status code of a path operation to 201?

*по-русски: Как задать код ответа 201 для операции пути?*

**Эталон.** Pass status_code to the path operation decorator; the status module provides named constants such as HTTP_201_CREATED.

**Ответ системы.**

```
You can set the status code of a path operation to 201 by using the `status_code` parameter in the path operation decorator. For example:

```Python
@app.post("/items/", status_code=201)
def create_item(item: Item):
    return item
```

This will return a 201 "Created" status code when the item is successfully created. [source: advanced__additional-status-codes.md]
```

Нужный документ в выдаче: **НЕТ**
Найдено: about__index.md, advanced__additional-responses.md, advanced__additional-status-codes.md
Ожидалось: tutorial__response-status-code.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 17. `q022` · подход `naive` · тип semantic

**Вопрос.** I want to answer with a rendered page built from a layout file, not with data. How?

*по-русски: Хочу отвечать отрисованной страницей из файла разметки, а не данными. Как?*

**Эталон.** Use Jinja2Templates: install jinja2, point Jinja2Templates at the directory, and return TemplateResponse from the path operation.

**Ответ системы.**

```
You can use templates to render a page. Write your template in a file, such as `templates/item.html`, and FastAPI will serve it as a rendered HTML response. [source: advanced__templates.md]
```

Нужный документ в выдаче: **да**
Найдено: advanced__custom-response.md, advanced__templates.md, how-to__general.md
Ожидалось: advanced__templates.md

<details><summary>вердикт судьи</summary>

**НЕВЕРНО**
</details>

Мой вердикт: `+`   ← расхождение с судьёй

---

## 18. `q012` · подход `hybrid` · тип lexical

**Вопрос.** What is APIRouter used for?

*по-русски: Для чего нужен APIRouter?*

**Эталон.** It lets you declare path operations in separate modules and then attach them to the main application with include_router.

**Ответ системы.**

```
`APIRouter` is used to declare *path operations* in a way similar to the `FastAPI` class, allowing for organization of routes and the inclusion of dependencies, responses, and other parameters to avoid code duplication. It can also be included in other routers or the main `FastAPI` application. [source: tutorial__bigger-applications.md]
```

Нужный документ в выдаче: **да**
Найдено: tutorial__bigger-applications.md, reference__apirouter.md, advanced__openapi-callbacks.md
Ожидалось: tutorial__bigger-applications.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## 19. `q020` · подход `reranked` · тип semantic

**Вопрос.** Many routes repeat identical setup. How do I declare that setup a single time?

*по-русски: Многие маршруты повторяют одинаковую подготовку. Как объявить её один раз?*

**Эталон.** Use the dependency injection system: write the shared logic as a function and declare it with Depends in every path operation that needs it.

**Ответ системы.**

```
NO DATA
```

Нужный документ в выдаче: **НЕТ**
Найдено: tutorial__dependencies__sub-dependencies.md, tutorial__bigger-applications.md, how-to__custom-request-and-route.md
Ожидалось: tutorial__dependencies__index.md

<details><summary>вердикт судьи</summary>

**НЕВЕРНО**
</details>

Мой вердикт: `-`

---

## 20. `q038` · подход `longctx_filtered` · тип multihop

**Вопрос.** How do I issue a JWT on login and then resolve the current user from it?

*по-русски: Как выдать JWT при входе и потом определить по нему текущего пользователя?*

**Эталон.** Verify the password on the token endpoint and sign a JWT, then decode that token inside a dependency and look up the user, depending on it in protected path operations.

**Ответ системы.**

```
To issue a JWT on login, you need to create a utility function that generates a new access token after verifying the user's credentials. Then, you can use a dependency like `get_current_user` to decode the received token, verify it, and return the current user. This process involves using the `oauth2_scheme` to extract the token and a utility function to validate it against your user model. 

For more details, refer to the sections on handling JWT tokens and getting the current user in the FastAPI documentation. [source: tutorial__security__oauth2-jwt.md]
```

Нужный документ в выдаче: **да**
Найдено: tutorial__security__oauth2-jwt.md, fastapi-people.md, tutorial__security__simple-oauth2.md
Ожидалось: tutorial__security__oauth2-jwt.md, tutorial__security__get-current-user.md

<details><summary>вердикт судьи</summary>

**ВЕРНО**
</details>

Мой вердикт: `+`

---

## Итог

**Совпало с судьёй: 17 из 20 → 85%.**

Расхождения на карточках 4, 14 и 17 — все три на расплывчатых ответах,
и они разнонаправленные: дважды судья мягче человека, один раз строже.
Систематического перекоса нет.

Критерий, по которому расходились: **ответ, отсылающий читателя
в документацию вместо ответа, ответом не считается.** Судья такие
засчитывает, человек — нет. Карточки 4 и 14 заканчиваются оборотами
вида «подробности см. в документации FastAPI»; карточка 17 короткая,
но конкретная, и потому засчитана.

Порог доверия был 15 из 20. Он пройден, поэтому колонка `correct`
используется в выводах — но как относительная величина для сравнения
подходов между собой, а не как абсолютная оценка качества.
