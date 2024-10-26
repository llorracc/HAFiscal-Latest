local domfilter = require "make4ht-domfilter"

local process = domfilter {
  function(dom)
    for _, img in ipairs(dom:query_selector("img")) do
      local src = img:get_attribute("src")
      if src then
        img:set_attribute("src", src:gsub("//", "/"))
      end
    end
    return dom
  end
}

Make:match("html$", process)