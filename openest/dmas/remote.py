def view_model(objtype, id):
  if objtype == 'url':
    while True:
      print("Accessing url...")
      try:
        return StoredModel.create_from_url(id, (float('-inf'), float('inf')))
    except Exception as ex:  # CATBELL
        import traceback; print("".join(traceback.format_exception(ex.__class__, ex, ex.__traceback__)))  # CATBELL
        time.sleep(1)
        pass

  if objtype == 'model':
    return MetaModel.get_model(id, False)

  if objtype == 'collection':
    collection = Collection.get(id, False)

    models = []
    for meta in collection.metas:
      models.append(meta.model())

    return Model.merge(models)

  if objtype == 'view':
    view = ModelView.get(id)
    return CombinationController.view_model(view.objtype, view.objid)
