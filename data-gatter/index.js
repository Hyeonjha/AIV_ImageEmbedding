import { GraphQLClient } from 'graphql-request';
import axios from 'axios';
import fs, { mkdirSync } from 'fs';
import sharp from 'sharp'
import path from 'path';
import { queryLabelSet, queryLabelSetIdsInVersion } from './graphql.js';

const endpoint = 'http://192.168.10.63:30832/graphql';

function getBbox(points, maxW, maxH, margin = 2,) {
  let minX = 9999999, minY = 9999999
  let maxX = -9999999, maxY = -9999999
  let w = 0, h = 0
  for (const point of points) {
    if (point.x < minX)
      minX = point.x
    if (point.y < minY)
      minY = point.y
    if (point.x > maxX)
      maxX = point.x
    if (point.y > maxY)
      maxY = point.y
  }
  w = maxX - minX
  h = maxY - minY

  let x = Math.round(minX - w / 2)
  let y = Math.round(minY - h / 2)
  w = w * margin
  h = h * margin

  // boundary exception handling
  x = x > 0 ? x : 0
  y = y > 0 ? y : 0
  w = x + w > maxW ? maxW - x : w
  h = y + h > maxH ? maxH - y : h

  return { x, y, w: w, h: h }
}
async function fetchImage(image, labelItems) {
  const url = `http://192.168.10.40:12010/api/v1/image?path=${image.path}`;

  try {
    const response = await axios({
      method: 'get',
      url: url,
      responseType: 'arraybuffer'
    });

    const imageBuffer = Buffer.from(response.data, 'binary');

    // 각 영역을 크롭하여 저장
    for (let i = 0; i < labelItems.length; i++) {
      const labelItem = labelItems[i];
      const outputDir = path.join(process.cwd(), 'downloaded', labelItem.LabelClass.name)
      mkdirSync(outputDir, { recursive: true })

      const region = getBbox(labelItem.points, image.width, image.height, 2)

      const outputFilePath = path.join(outputDir, `${labelItem.id}.jpg`);

      await sharp(imageBuffer)
        .extract({ left: region.x, top: region.y, width: region.w, height: region.h })
        .toFile(outputFilePath);
    }
  } catch (error) {
    console.error('Error fetching image', error);
  }
}


async function fetchGraphQLData() {
  const graphQLClient = new GraphQLClient(endpoint, {
    headers: {
    },
  });

  try {
    const { labelSetIdsInVersion } = await graphQLClient.request(queryLabelSetIdsInVersion);

    for (const labelSetId of labelSetIdsInVersion) {
      const data2 = await graphQLClient.request(queryLabelSet, { id: labelSetId });

      fetchImage(data2.labelSet.Image, data2.labelSet.LabelItem);
    }
  } catch (error) {
    console.error(error);
  }
}

fetchGraphQLData();
